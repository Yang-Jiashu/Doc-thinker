export interface VisualDepthResult {
  depths: Float32Array;
  maximumDepth: number;
}

export function computeVisualDepths(
  positions: Float32Array<ArrayBufferLike>,
  hashes: Uint32Array<ArrayBufferLike>,
): VisualDepthResult {
  const count = Math.min(hashes.length, Math.floor(positions.length / 3));
  const depths = new Float32Array(count);
  if (!count) return { depths, maximumDepth: 0 };

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < count; index += 1) {
    minX = Math.min(minX, positions[index * 3]);
    maxX = Math.max(maxX, positions[index * 3]);
    minY = Math.min(minY, positions[index * 3 + 1]);
    maxY = Math.max(maxY, positions[index * 3 + 1]);
  }

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  let maximumRadius = 1;
  for (let index = 0; index < count; index += 1) {
    maximumRadius = Math.max(
      maximumRadius,
      Math.hypot(positions[index * 3] - centerX, positions[index * 3 + 1] - centerY),
    );
  }
  const maximumDepth = maximumRadius * 0.34;

  for (let index = 0; index < count; index += 1) {
    const x = positions[index * 3] - centerX;
    const y = positions[index * 3 + 1] - centerY;
    const radialRatio = Math.min(1, Math.hypot(x, y) / maximumRadius);
    const availableDepth = Math.sqrt(Math.max(0, 1 - radialRatio * radialRatio)) * maximumDepth;
    const hash = hashes[index] >>> 0;
    const sign = (hash & 1) === 0 ? -1 : 1;
    const layer = 0.38 + ((hash >>> 8) & 0xffff) / 0xffff * 0.62;
    depths[index] = sign * availableDepth * layer;
  }

  return { depths, maximumDepth };
}
