import { mkdir } from "node:fs/promises";
import path from "node:path";
import { readdir, rename, copyFile, writeFile } from "node:fs/promises";

export const SessionFolderPlugin = async ({ directory, client }) => {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const baseDir = path.resolve(directory, ".opencode_sessions");
  const sessionDir = path.join(baseDir, timestamp);
  const sessionStartTime = Date.now();
  
  try {
    await mkdir(baseDir, { recursive: true });
    await mkdir(sessionDir, { recursive: true });
    
    // Create a link file to track the session
    const linkFile = path.join(baseDir, "current_session.txt");
    await writeFile(linkFile, sessionDir, "utf8");
    
    // Register this session as the current active one
    global._activeSession = {
      id: timestamp,
      dir: sessionDir,
      startTime: sessionStartTime,
      processed: new Set()
    };
    
    await client.app.log({
      body: {
        service: "session-folder-plugin",
        level: "info",
        message: `Session active: ${timestamp}`
      }
    });
    
    // Start polling for export files (only once globally)
    if (!global._exportPollStarted) {
      global._exportPollStarted = true;
      
      setInterval(async () => {
        try {
          // Check if there's an active session
          if (!global._activeSession) return;
          
          const files = await readdir(directory);
          
          for (const file of files) {
            // Check if it's a session export file that hasn't been processed
            if (file.startsWith('session-') && file.endsWith('.md')) {
              // Only process if this is the active session and hasn't been processed
              if (!global._activeSession.processed.has(file)) {
                global._activeSession.processed.add(file);
                
                const sourcePath = path.join(directory, file);
                const destPath = path.join(global._activeSession.dir, file);
                
                await client.app.log({
                  body: {
                    service: "session-folder-plugin",
                    level: "info",
                    message: `Moving export`,
                    extra: { 
                      file,
                      session: global._activeSession.id
                    }
                  }
                });
                
                try {
                  await rename(sourcePath, destPath);
                } catch (e) {
                  try {
                    await copyFile(sourcePath, destPath);
                  } catch (e2) {
                    await client.app.log({
                      body: {
                        service: "session-folder-plugin",
                        level: "error",
                        message: `Failed to move ${file}: ${e2.message}`
                      }
                    });
                  }
                }
              }
            }
          }
        } catch (e) {
          // Silent fail on polling errors
        }
      }, 1000); // Check every second
    }
    
  } catch (error) {
    console.error("[SessionFolderPlugin] Error:", error);
  }
  
  return {};
};