import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { TOKEN_FILE_DIR, TOKEN_FILE_NAME } from '../constants.js';
import type { OAuthTokens } from '../types.js';

function getTokenFilePath(): string {
  return path.join(os.homedir(), TOKEN_FILE_DIR, TOKEN_FILE_NAME);
}

function ensureDir(filePath: string): void {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  }
}

export function loadTokens(): OAuthTokens | null {
  const filePath = getTokenFilePath();
  try {
    if (!fs.existsSync(filePath)) return null;
    const raw = fs.readFileSync(filePath, 'utf-8');
    const tokens = JSON.parse(raw) as OAuthTokens;
    if (!tokens.access_token || !tokens.refresh_token) return null;
    return tokens;
  } catch {
    return null;
  }
}

export function saveTokens(tokens: OAuthTokens): void {
  const filePath = getTokenFilePath();
  ensureDir(filePath);
  fs.writeFileSync(filePath, JSON.stringify(tokens, null, 2), {
    encoding: 'utf-8',
    mode: 0o600,
  });
}

export function clearTokens(): void {
  const filePath = getTokenFilePath();
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch {
    // Ignore errors during cleanup
  }
}
