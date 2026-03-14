import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { getSessions } from '../api/kubix-client.js';
import { sanitizeError } from '../types.js';

export function registerKubixSessions(server: McpServer): void {
  server.registerTool(
    'kubix_sessions',
    {
      title: 'List Optimization Sessions',
      description:
        'List past Kubix optimization sessions with their results. Useful for checking what has already been optimized.',
      inputSchema: {
        limit: z
          .number()
          .min(1)
          .max(100)
          .default(10)
          .describe('Number of sessions to return'),
        status: z
          .enum(['all', 'completed', 'failed', 'running'])
          .default('all')
          .describe('Filter by session status'),
      },
    },
    async ({ limit, status }) => {
      try {
        const result = await getSessions(limit, status);

        if (result.sessions.length === 0) {
          return {
            content: [
              {
                type: 'text',
                text: 'No optimization sessions found. Use kubix_optimize to start one.',
              },
            ],
          };
        }

        const lines = [
          `## Optimization Sessions (${result.sessions.length} of ${result.total})`,
          '',
          '| Task | GPU | Speedup | Status | Format | Created |',
          '|------|-----|---------|--------|--------|---------|',
        ];

        for (const s of result.sessions) {
          const speedup = s.best_speedup ? `${s.best_speedup.toFixed(2)}x` : '-';
          const date = new Date(s.created_at).toLocaleDateString();
          lines.push(
            `| ${s.task_name} | ${s.gpu_type || '-'} | ${speedup} | ${s.status} | ${s.output_format || '-'} | ${date} |`
          );
        }

        if (result.has_more) {
          lines.push('', `_Showing ${result.sessions.length} of ${result.total} sessions._`);
        }

        return { content: [{ type: 'text', text: lines.join('\n') }] };
      } catch (err) {
        return {
          content: [{ type: 'text', text: `Failed to list sessions: ${sanitizeError(err)}` }],
          isError: true,
        };
      }
    }
  );
}
