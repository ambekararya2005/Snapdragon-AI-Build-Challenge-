// Render stills at given times: node stills.cjs out_dir t1 t2 ...
const { chromium } = require('playwright');
const path = require('node:path');
const fs = require('node:fs');
(async () => {
  const [, , outDir, ...times] = process.argv;
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1350 }, deviceScaleFactor: 1 });
  page.on('console', (m) => console.log('console:', m.text()));
  page.on('pageerror', (e) => console.log('pageerror:', e.message));
  await page.goto('file://' + path.resolve('composition/index.html'));
  await page.evaluate(() => window.ready);
  for (const t of times) {
    await page.evaluate((tt) => window.renderAt(tt), +t);
    await page.screenshot({ path: path.join(outDir, `t${(+t).toFixed(2).padStart(6, '0')}.png`) });
  }
  await browser.close();
})();
