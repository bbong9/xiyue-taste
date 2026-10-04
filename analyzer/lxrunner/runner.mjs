// Xiyue container source runner: one vm context per enabled source, running the same
// user-api-preload.js as iOS. The natives, the load order and the request/response
// delivery follow iOS LXMusicWebWorkerRuntime.
// vm is not a security boundary; the permission flags are the backstop: Node may only
// read this directory, and may not write files, start processes or start workers.
// Node never touches the network: a script's lx.request becomes a request line, and
// Python makes the request by the iOS network rules and answers with a response line.
//
// Protocol: one JSON object per line on stdin/stdout. stderr carries only "LXRUNNER ..."
// diagnostics, never script text or addresses. Node's lines are at most 1 MiB of UTF-8.
// Python → Node
//   {"type":"load","id","script","meta":{"name","description","version","author","homepage"},"timeoutMs"}
//   {"type":"unload","id"}
//   {"type":"call","callId","id","source","quality","musicInfo","timeoutMs"}
//   {"type":"response","id","requestKey","errorCode","responseJSON","bodyText"}
// Node → Python
//   {"type":"ready"}
//   {"type":"loaded","id","ok":true,"sources":{"kw":["128k",...]}} / {"type":"loaded","id","ok":false,"error"}
//   {"type":"request","id","requestKey","url","options"}
//   {"type":"cancel","id","requestKey"}
//   {"type":"result","callId","ok":true,"url"} / {"type":"result","callId","ok":false,"error"}
//   {"type":"alert","id"}

import crypto from 'node:crypto'
import fs from 'node:fs'
import readline from 'node:readline'
import vm from 'node:vm'

const PRELOAD = fs.readFileSync(new URL('./user-api-preload.js', import.meta.url), 'utf8')
const PLATFORMS = ['kw', 'kg', 'tx', 'wy', 'mg']
const QUALITIES = ['128k', '320k', 'flac', 'flac24bit', 'hires', 'atmos', 'atmos_plus', 'master']
const MAX_INIT_INFO_BYTES = 64 * 1024
const MAX_BRIDGE_RESPONSE_BYTES = 10 * 1024 * 1024
const MAX_ALERT_BYTES = 8 * 1024
const MAX_TIMER_MS = 2147483647
const MAX_LINE_BYTES = 1024 * 1024

const runtimes = new Map()
const calls = new Map()

const send = message => {
  process.stdout.write(JSON.stringify(message) + '\n')
}

const diagnose = text => {
  process.stderr.write('LXRUNNER ' + text + '\n')
}

const utf8Length = value => Buffer.byteLength(value, 'utf8')

// iOS measures with sortedKeys; key order does not change the length.
const jsonLength = value => {
  try {
    const text = JSON.stringify(value)
    return text === undefined ? -1 : utf8Length(text)
  } catch {
    return -1
  }
}

// atob's forgiving-base64: drop whitespace; when the length is a multiple of 4 drop one or two
// trailing '='; a remainder of 1 or a character outside the alphabet is invalid.
const forgivingBase64 = input => {
  let text = input.replace(/[\t\n\f\r ]/g, '')
  if (text.length % 4 === 0) text = text.replace(/==?$/, '')
  if (text.length % 4 === 1 || /[^A-Za-z0-9+/]/.test(text)) return null
  return Buffer.from(text, 'base64')
}

// These host functions run outside the vm. They take and return primitives only and swallow
// every exception, so no outer object leaks into the vm.
const hostStr2b64 = input => {
  try {
    return Buffer.from(String(input), 'utf8').toString('base64')
  } catch {
    return ''
  }
}

const hostB642buf = input => {
  try {
    const bytes = forgivingBase64(String(input))
    return bytes === null ? null : JSON.stringify(Array.from(bytes))
  } catch {
    return null
  }
}

const hostMd5 = input => {
  try {
    let text = String(input)
    try {
      text = decodeURIComponent(text)
    } catch {}
    return crypto.createHash('md5').update(text, 'utf8').digest('hex')
  } catch {
    return ''
  }
}

// As upstream Android utils_aes_encrypt: 'AES' is Java's default AES/ECB/PKCS5Padding; the CBC
// IV is zero-padded or cut to 16 bytes, and a random one is used when none is given; a 16/24/32
// byte key picks AES-128/192/256; any error gives an empty string.
const hostAes = (data, key, iv, mode) => {
  try {
    const input = Buffer.from(String(data), 'base64')
    const keyBytes = Buffer.from(String(key), 'base64')
    const bits = { 16: 128, 24: 192, 32: 256 }[keyBytes.length]
    if (!bits) return ''
    let cipher
    if (mode === 'AES/CBC/PKCS7Padding') {
      let ivBytes = Buffer.from(String(iv), 'base64')
      if (ivBytes.length === 0) ivBytes = crypto.randomBytes(16)
      const fixed = Buffer.alloc(16)
      ivBytes.copy(fixed, 0, 0, Math.min(16, ivBytes.length))
      cipher = crypto.createCipheriv(`aes-${bits}-cbc`, keyBytes, fixed)
    } else if (mode === 'AES') {
      cipher = crypto.createCipheriv(`aes-${bits}-ecb`, keyBytes, null)
    } else {
      return ''
    }
    return Buffer.concat([cipher.update(input), cipher.final()]).toString('base64')
  } catch {
    return ''
  }
}

// As upstream Android utils_rsa_encrypt: the key is base64 SPKI without the PEM lines; with
// NoPadding the data is left-padded with zeros to the modulus length.
const hostRsa = (data, key, padding) => {
  try {
    if (padding !== 'RSA/ECB/NoPadding') return ''
    const der = Buffer.from(String(key).replace(/\s+/g, ''), 'base64')
    const publicKey = crypto.createPublicKey({ key: der, format: 'der', type: 'spki' })
    const size = Math.ceil(publicKey.asymmetricKeyDetails.modulusLength / 8)
    const input = Buffer.from(String(data), 'base64')
    if (input.length > size) return ''
    const padded = Buffer.alloc(size)
    input.copy(padded, size - input.length)
    return crypto.publicEncrypt({ key: publicKey, padding: crypto.constants.RSA_NO_PADDING }, padded).toString('base64')
  } catch {
    return ''
  }
}

// Installs the iOS __lx_native_call__* globals in the vm. The outer functions reach the vm only
// through this closure, never as globals; b642buf throws the vm's own Error when the input does
// not decode, as atob does on iOS.
const INSTALL_NATIVES = `(function (hostCall, hostTimer, hostStr2b64, hostB642buf, hostMd5, hostAes, hostRsa) {
  'use strict'
  const text = value => String(value ?? '')
  globalThis.__lx_native_call__ = (key, action, data) => {
    hostCall(String(key), String(action), data == null ? null : String(data))
  }
  globalThis.__lx_native_call__set_timeout = (id, timeout) => {
    hostTimer(typeof id === 'number' ? id : NaN, typeof timeout === 'number' ? timeout : NaN)
  }
  globalThis.__lx_native_call__utils_str2b64 = input => hostStr2b64(text(input))
  globalThis.__lx_native_call__utils_b642buf = input => {
    const result = hostB642buf(text(input))
    if (typeof result !== 'string') throw new Error('The string to be decoded is not correctly encoded.')
    return result
  }
  globalThis.__lx_native_call__utils_str2md5 = input => hostMd5(text(input))
  globalThis.__lx_native_call__utils_aes_encrypt = (data, key, iv, mode) => hostAes(text(data), text(key), text(iv), text(mode))
  globalThis.__lx_native_call__utils_rsa_encrypt = (data, key, padding) => hostRsa(text(data), text(key), text(padding))
})`

// iOS parseSupports: only kw/wy/tx/kg/mg platforms that declare musicUrl.
const parseSupports = info => {
  if (info == null || typeof info !== 'object' || Array.isArray(info)) return null
  const sources = info.sources
  if (sources == null || typeof sources !== 'object' || Array.isArray(sources)) return null
  const parsed = {}
  for (const platform of PLATFORMS) {
    const raw = sources[platform]
    if (raw == null || typeof raw !== 'object' || Array.isArray(raw)) continue
    if (raw.type !== 'music' || !Array.isArray(raw.actions) || !Array.isArray(raw.qualitys)) continue
    if (!raw.actions.includes('musicUrl')) continue
    const qualities = []
    for (const value of raw.qualitys) {
      if (typeof value !== 'string') continue
      const quality = value.trim().toLowerCase()
      if (QUALITIES.includes(quality) && !qualities.includes(quality)) qualities.push(quality)
    }
    parsed[platform] = qualities
  }
  return Object.keys(parsed).length ? parsed : null
}

class Runtime {
  constructor(id, script, meta, timeoutMs) {
    this.id = id
    this.key = crypto.randomUUID()
    this.timers = new Map()
    this.pending = new Map()
    this.invocations = new Map()
    this.initializing = true
    this.evaluated = false
    this.initOutcome = null
    this.unloaded = false
    this.native = null
    this.load(script, meta, timeoutMs)
  }

  // The iOS order: natives, preload, lx_setup, the script, then wait for init. A synchronous
  // throw from the script fails the load even when it has already sent init.
  load(script, meta, timeoutMs) {
    this.initDeadline = setTimeout(() => this.finishLoad(false, 'initialization_timeout'), timeoutMs)
    const started = performance.now()
    try {
      this.context = vm.createContext(Object.create(null))
      vm.runInContext(INSTALL_NATIVES, this.context, { timeout: 1000 })(
        (key, action, data) => this.nativeCall(key, action, data),
        (id, timeout) => this.setTimer(id, timeout),
        hostStr2b64, hostB642buf, hostMd5, hostAes, hostRsa,
      )
      new vm.Script(PRELOAD, { filename: 'user-api-preload.js' }).runInContext(this.context, { timeout: timeoutMs })
      const field = (name, fallback) => (typeof meta?.[name] === 'string' ? meta[name] : fallback)
      vm.runInContext('globalThis.lx_setup', this.context)(
        this.key, this.id, field('name', 'Unknown'), field('description', ''), field('version', ''),
        field('author', ''), field('homepage', ''), script,
      )
      this.native = vm.runInContext('globalThis.__lx_native__', this.context)
      new vm.Script(script, { filename: 'source.js' }).runInContext(this.context, { timeout: timeoutMs })
    } catch {
      // Never read what the script threw (a getter would run outside the vm timeout); tell a
      // timeout by the time taken.
      const timedOut = performance.now() - started >= timeoutMs
      this.dispatch('__run_error__', '{}')
      this.finishLoad(false, timedOut ? 'initialization_timeout' : 'script_failed')
      return
    }
    this.evaluated = true
    if (this.initOutcome) this.finishLoad(...this.initOutcome)
  }

  finishLoad(ok, value) {
    if (!this.initializing) return
    this.initializing = false
    clearTimeout(this.initDeadline)
    if (ok) {
      send({ type: 'loaded', id: this.id, ok: true, sources: value })
    } else {
      send({ type: 'loaded', id: this.id, ok: false, error: value })
      this.unload()
      if (runtimes.get(this.id) === this) runtimes.delete(this.id)
    }
  }

  // Calls the vm's frozen __lx_native__; nothing it throws gets out.
  dispatch(action, data) {
    if (this.unloaded || typeof this.native !== 'function') return
    try {
      this.native(this.key, action, data)
    } catch {}
  }

  nativeCall(key, action, data) {
    if (this.unloaded || key !== this.key) return
    switch (action) {
      case 'init': return this.handleInit(data)
      case 'request': return this.handleRequest(data)
      case 'cancelRequest': return this.handleCancelRequest(data)
      case 'response': return this.handleResponse(data)
      case 'showUpdateAlert': return this.handleUpdateAlert(data)
    }
  }

  handleInit(data) {
    if (!this.initializing || this.initOutcome) return
    let object
    try {
      object = JSON.parse(data)
    } catch {
      object = null
    }
    const infoLength = object?.info == null ? -1 : jsonLength(object.info)
    if (object == null || object.status !== true || object.info == null) {
      this.initOutcome = [false, 'script_failed']
    } else if (infoLength < 0 || infoLength > MAX_INIT_INFO_BYTES) {
      this.initOutcome = [false, 'protocol_violation']
    } else {
      const sources = parseSupports(object.info)
      this.initOutcome = sources === null ? [false, 'initialization_failed'] : [true, sources]
    }
    if (this.evaluated) this.finishLoad(...this.initOutcome)
  }

  handleUpdateAlert(data) {
    if (data == null || utf8Length(data) > MAX_ALERT_BYTES) return
    try {
      const object = JSON.parse(data)
      if (typeof object?.log !== 'string' || !object.log.trim()) return
    } catch {
      return
    }
    send({ type: 'alert', id: this.id })
  }

  setTimer(id, timeout) {
    if (this.unloaded || !Number.isInteger(id) || id < 0 || !Number.isInteger(timeout) || timeout < 0) return
    clearTimeout(this.timers.get(id))
    this.timers.set(id, setTimeout(() => {
      this.timers.delete(id)
      this.dispatch('__set_timeout__', String(id))
    }, Math.min(timeout, MAX_TIMER_MS)))
  }

  handleRequest(data) {
    let object
    try {
      object = JSON.parse(data)
    } catch {
      return
    }
    if (object == null || typeof object.requestKey !== 'string' || typeof object.url !== 'string' || object.options === undefined) return
    const { requestKey, url, options } = object
    // Sizes, concurrency and addresses are checked by Python, the trust boundary. Here only a
    // key already waiting, or a line too long for the pipe, is turned away.
    const line = JSON.stringify({ type: 'request', id: this.id, requestKey, url, options })
    if (this.pending.has(requestKey) || line === undefined || utf8Length(line) > MAX_LINE_BYTES) {
      setImmediate(() => this.deliverResponse(requestKey, 'request_rejected', '{}', '', false))
      return
    }
    this.pending.set(requestKey, { binary: options?.binary === true, cancelled: false })
    process.stdout.write(line + '\n')
  }

  handleCancelRequest(data) {
    let requestKey = null
    try {
      const value = JSON.parse(data)
      requestKey = typeof value === 'string' ? value : value?.requestKey
    } catch {}
    const pending = typeof requestKey === 'string' ? this.pending.get(requestKey) : null
    if (!pending || pending.cancelled) return
    pending.cancelled = true
    send({ type: 'cancel', id: this.id, requestKey })
  }

  networkReply({ requestKey, errorCode, responseJSON, bodyText }) {
    const pending = this.pending.get(requestKey)
    if (!pending) return
    this.pending.delete(requestKey)
    if (pending.cancelled) {
      this.deliverResponse(requestKey, 'cancelled', '{}', '', pending.binary)
      return
    }
    const code = typeof errorCode === 'string' ? errorCode : 'unavailable'
    const json = typeof responseJSON === 'string' ? responseJSON : '{}'
    const text = typeof bodyText === 'string' ? bodyText : ''
    if (utf8Length(code) + utf8Length(json) + utf8Length(text) > MAX_BRIDGE_RESPONSE_BYTES) {
      this.deliverResponse(requestKey, 'response_too_large', '{}', '', pending.binary)
      return
    }
    this.deliverResponse(requestKey, code, json, text, pending.binary)
  }

  // iOS deliverResponse: binary takes response.body; text is parsed as JSON (fragments too) and
  // falls back to the raw text.
  deliverResponse(requestKey, errorCode, responseJSON, bodyText, binary) {
    let response
    try {
      response = JSON.parse(responseJSON)
    } catch {}
    if (response == null || typeof response !== 'object' || Array.isArray(response)) response = {}
    let body
    if (binary) {
      body = response.body ?? bodyText
    } else {
      try {
        body = JSON.parse(bodyText)
      } catch {
        body = bodyText
      }
    }
    response.body = body
    this.dispatch('response', JSON.stringify({ requestKey, error: errorCode || null, response }))
  }

  invoke(callId, source, quality, musicInfo, timeoutMs) {
    if (this.initializing || this.unloaded) {
      finishCall(callId, false, 'unavailable')
      return
    }
    const requestKey = 'call-' + callId
    const deadline = setTimeout(() => this.finishInvocation(requestKey, false, 'timeout'), timeoutMs)
    this.invocations.set(requestKey, { callId, deadline })
    try {
      this.native(this.key, 'request', JSON.stringify({
        requestKey,
        data: { source, action: 'musicUrl', info: { type: quality, musicInfo } },
      }))
    } catch {
      this.finishInvocation(requestKey, false, 'script_failed')
    }
  }

  // iOS handleResponse: needs status === true and result.data; takes data.url when data is an
  // object with a string url.
  handleResponse(data) {
    let object
    try {
      object = JSON.parse(data)
    } catch {
      return
    }
    if (typeof object?.requestKey !== 'string' || !this.invocations.has(object.requestKey)) return
    const result = object.result
    if (object.status !== true || result == null || typeof result !== 'object' || result.data === undefined) {
      this.finishInvocation(object.requestKey, false, 'script_failed')
      return
    }
    const value = result.data
    if (value != null && typeof value === 'object' && typeof value.url === 'string') {
      this.finishInvocation(object.requestKey, true, value.url)
    } else {
      this.finishInvocation(object.requestKey, false, 'invalid_result')
    }
  }

  finishInvocation(requestKey, ok, value) {
    const invocation = this.invocations.get(requestKey)
    if (!invocation) return
    this.invocations.delete(requestKey)
    clearTimeout(invocation.deadline)
    finishCall(invocation.callId, ok, value)
  }

  unload() {
    if (this.unloaded) return
    this.unloaded = true
    clearTimeout(this.initDeadline)
    for (const timer of this.timers.values()) clearTimeout(timer)
    this.timers.clear()
    this.pending.clear()
    for (const requestKey of [...this.invocations.keys()]) this.finishInvocation(requestKey, false, 'unavailable')
    this.native = null
    this.context = null
  }
}

// An address too long for a line is an invalid result; Python checks the rest.
const finishCall = (callId, ok, value) => {
  if (!calls.delete(callId)) return
  const line = ok ? JSON.stringify({ type: 'result', callId, ok: true, url: value }) : null
  if (line !== null && utf8Length(line) <= MAX_LINE_BYTES) {
    process.stdout.write(line + '\n')
  } else {
    send({ type: 'result', callId, ok: false, error: ok ? 'invalid_result' : value })
  }
}

const handle = message => {
  switch (message.type) {
    case 'load': {
      runtimes.get(message.id)?.unload()
      runtimes.delete(message.id)
      const timeoutMs = Number.isInteger(message.timeoutMs) && message.timeoutMs > 0 ? message.timeoutMs : 12000
      const runtime = new Runtime(String(message.id), String(message.script), message.meta, timeoutMs)
      if (!runtime.unloaded) runtimes.set(runtime.id, runtime)
      return
    }
    case 'unload':
      runtimes.get(message.id)?.unload()
      runtimes.delete(message.id)
      return
    case 'call': {
      const callId = String(message.callId)
      calls.set(callId, true)
      const runtime = runtimes.get(message.id)
      if (!runtime) {
        finishCall(callId, false, 'unavailable')
        return
      }
      const timeoutMs = Number.isInteger(message.timeoutMs) && message.timeoutMs > 0 ? message.timeoutMs : 35000
      runtime.invoke(callId, String(message.source), String(message.quality), message.musicInfo, timeoutMs)
      return
    }
    case 'response':
      runtimes.get(message.id)?.networkReply(message)
  }
}

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity })
lines.on('line', line => {
  let message
  try {
    message = JSON.parse(line)
  } catch {
    diagnose('bad-message')
    return
  }
  if (message == null || typeof message !== 'object') return
  try {
    handle(message)
  } catch {
    diagnose('handler-error type=' + String(message.type).slice(0, 20))
  }
})
lines.on('close', () => process.exit(0))
process.on('uncaughtException', () => {
  diagnose('uncaught')
})
// A rejection nobody handles inside a script neither ends the process nor gets logged.
process.on('unhandledRejection', () => {})
send({ type: 'ready' })
