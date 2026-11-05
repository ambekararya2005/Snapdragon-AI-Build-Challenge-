// Render every frame (pure function of time) and pipe JPEGs into ffmpeg: node render.cjs out.mp4
const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const path = require('node:path');
(async () => {
  const out = process.argv[2] || 'video.mp4';
  const FPS = 30, DUR = 38, NF = FPS * DUR;
  const ff = spawn('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'mjpeg', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-r', String(FPS), out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1350 }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.log('pageerror:', e.message));
  await page.goto('file://' + path.resolve('composition/index.html'), { waitUntil: 'load' });
  await page.evaluate(() => window.ready);
  const t0 = Date.now();
  for (let i = 0; i < NF; i++) {
    await page.evaluate((tt) => window.renderAt(tt), i / FPS);
    const buf = await page.screenshot({ type: 'jpeg', quality: 95 });
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    if (i % 150 === 0) console.log(`frame ${i}/${NF} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  await browser.close();
  console.log('done', out, ((Date.now() - t0) / 1000).toFixed(0) + 's');
})();
