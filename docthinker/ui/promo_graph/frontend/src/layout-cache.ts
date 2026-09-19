interface LayoutRecord {
  key: string;
  nodeCount: number;
  positions: ArrayBuffer;
  savedAt: number;
}

const MEMORY_CACHE = new Map<string, Float32Array>();
const DB_NAME = "docthinker-promo-layouts";
const STORE_NAME = "layouts";

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME, { keyPath: "key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export class LayoutCache {
  async get(key: string, nodeCount: number): Promise<Float32Array | null> {
    const memory = MEMORY_CACHE.get(key);
    if (memory?.length === nodeCount * 3) return memory.slice();
    if (typeof indexedDB === "undefined") return null;
    try {
      const database = await openDatabase();
      const record = await new Promise<LayoutRecord | undefined>((resolve, reject) => {
        const transaction = database.transaction(STORE_NAME, "readonly");
        const request = transaction.objectStore(STORE_NAME).get(key);
        request.onsuccess = () => resolve(request.result as LayoutRecord | undefined);
        request.onerror = () => reject(request.error);
      });
      database.close();
      if (!record || record.nodeCount !== nodeCount || record.positions.byteLength !== nodeCount * 3 * 4) return null;
      const positions = new Float32Array(record.positions.slice(0));
      MEMORY_CACHE.set(key, positions.slice());
      return positions;
    } catch {
      return null;
    }
  }

  async set(key: string, positions: Float32Array): Promise<void> {
    MEMORY_CACHE.set(key, positions.slice());
    if (typeof indexedDB === "undefined") return;
    try {
      const database = await openDatabase();
      await new Promise<void>((resolve, reject) => {
        const transaction = database.transaction(STORE_NAME, "readwrite");
        transaction.objectStore(STORE_NAME).put({
          key,
          nodeCount: positions.length / 3,
          positions: positions.slice().buffer as ArrayBuffer,
          savedAt: Date.now(),
        } satisfies LayoutRecord);
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
      });
      database.close();
    } catch {
      // IndexedDB is an optimization; rendering continues when it is unavailable.
    }
  }
}

export function isLayoutCompatible(positions: Float32Array | null, nodeCount: number): boolean {
  return Boolean(positions && positions.length === nodeCount * 3);
}
