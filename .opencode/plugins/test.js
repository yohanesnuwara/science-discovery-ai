// Test plugin to verify session ID
import { mkdir } from "node:fs/promises";
import path from "node:path";

export const TestPlugin = async ({ client, directory }) => {
  const sessionId = process.env.OPEN_CODE_SESSION_ID || process.env.SESSION_ID || "unknown-session";
  const baseDir = path.resolve(directory, ".opencode_sessions");
  
  try {
    await client.app.log({
      body: {
        service: "test-plugin",
        level: "info",
        message: "Plugin loaded",
        extra: { sessionId }
      }
    });
    
    // Create folder immediately
    const sessionDir = path.join(baseDir, String(sessionId));
    await mkdir(baseDir, { recursive: true });
    await mkdir(sessionDir, { recursive: true });
    
    await client.app.log({
      body: {
        service: "test-plugin",
        level: "info",
        message: `Created folder: ${sessionDir}`
      }
    });
  } catch (error) {
    console.log(`Plugin error: ${error.message}`);
  }
  
  return {};
};
