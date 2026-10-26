'use strict';

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const signature = Buffer.from('89504e470d0a1a0a', 'hex');

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const file = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(file) : /\.png$/i.test(file) ? [file] : [];
  });
}

let saved = 0;
let changed = 0;
for (const file of walk('source')) {
  const original = fs.readFileSync(file);
  if (!original.subarray(0, 8).equals(signature)) throw new Error(`Invalid PNG: ${file}`);
  const chunks = [];
  for (let offset = 8; offset < original.length;) {
    const length = original.readUInt32BE(offset);
    const end = offset + length + 12;
    if (end > original.length) throw new Error(`Truncated PNG: ${file}`);
    chunks.push({ type: original.toString('ascii', offset + 4, offset + 8), bytes: original.subarray(offset, end) });
    offset = end;
  }
  const raw = zlib.inflateSync(Buffer.concat(chunks.filter(c => c.type === 'IDAT').map(c => c.bytes.subarray(8, -4))));
  const alternatives = [zlib.constants.Z_DEFAULT_STRATEGY, zlib.constants.Z_FILTERED].map(strategy =>
    zlib.deflateSync(raw, { level: 9, memLevel: 9, strategy }));
  const compressed = alternatives.reduce((a, b) => a.length < b.length ? a : b);
  if (!zlib.inflateSync(compressed).equals(raw)) throw new Error(`Lossless check failed: ${file}`);
  const idat = Buffer.alloc(compressed.length + 12);
  idat.writeUInt32BE(compressed.length, 0);
  idat.write('IDAT', 4);
  compressed.copy(idat, 8);
  idat.writeUInt32BE(crc32(idat.subarray(4, -4)), idat.length - 4);
  let inserted = false;
  const result = Buffer.concat([signature, ...chunks.flatMap(chunk => {
    if (chunk.type !== 'IDAT') return [chunk.bytes];
    if (inserted) return [];
    inserted = true;
    return [idat];
  })]);
  // Every non-image-data chunk (including color profiles) stays byte-identical.
  if (result.length < original.length) {
    fs.writeFileSync(file, result);
    changed++;
    saved += original.length - result.length;
  }
}
console.log(`Losslessly optimized ${changed} PNG files; saved ${saved} bytes. Dimensions and metadata are unchanged.`);
