import { SSE_TIMEOUT_MS } from '../constants.js';
import type { SSEEvent } from '../types.js';

/**
 * Consume an SSE stream using native fetch + ReadableStream.
 * Blocks until session_complete, error, or timeout.
 * Returns all collected events.
 */
export async function consumeSSEStream(
  url: string,
  token: string,
  onEvent?: (event: SSEEvent) => void
): Promise<SSEEvent[]> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), SSE_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'text/event-stream',
        'Cache-Control': 'no-cache',
      },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`SSE connection failed: HTTP ${response.status}`);
    }

    if (!response.body) {
      throw new Error('SSE response has no body');
    }

    const events: SSEEvent[] = [];
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE frames — frames are separated by double newlines
      const frames = buffer.split('\n\n');
      // Keep the last incomplete frame in the buffer
      buffer = frames.pop() || '';

      for (const frame of frames) {
        const event = parseSSEFrame(frame);
        if (!event) continue;

        events.push(event);

        // Protect caller from callback crashes
        if (onEvent) {
          try {
            onEvent(event);
          } catch {
            // Callback error must not kill the stream
          }
        }

        // Terminal events
        if (event.type === 'session_complete' || event.type === 'error') {
          reader.cancel();
          return events;
        }
      }
    }

    return events;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Parse a single SSE frame into an SSEEvent.
 * SSE format:
 *   event: <type>
 *   data: <json>
 *   id: <id>
 *
 * Per the SSE spec, multiple `data:` lines are joined with newlines.
 */
function parseSSEFrame(frame: string): SSEEvent | null {
  const lines = frame.split('\n');
  let eventType = '';
  const dataLines: string[] = [];
  let id = '';

  for (const line of lines) {
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim());
    } else if (line.startsWith('id:')) {
      id = line.slice(3).trim();
    }
  }

  if (dataLines.length === 0) return null;

  // Per SSE spec, multiple data lines are joined by newlines
  const dataStr = dataLines.join('\n');

  try {
    const parsed = JSON.parse(dataStr);

    // Some servers wrap the event in an envelope with id/type/data/timestamp
    if (parsed.type && parsed.data && typeof parsed.data === 'object') {
      return {
        id: parsed.id || id || '',
        type: parsed.type,
        data: parsed.data,
        timestamp: parsed.timestamp || new Date().toISOString(),
      };
    }

    // Raw data without envelope
    return {
      id: id || '',
      type: (eventType || 'message') as SSEEvent['type'],
      data: parsed,
      timestamp: new Date().toISOString(),
    };
  } catch {
    return null;
  }
}
