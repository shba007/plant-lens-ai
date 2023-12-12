import puppeteer from 'puppeteer';
import { downloadFile } from "./utils.js";

const plants = [
	// "marigold",
	// "jasmine",
	// "rose",
	// "hibiscus",
	// "sunflower",
	// "dahlia",
	// "lotus",
	// "bougainvillea",
	// "chrysanthemum",
	// "lily",
	// "lavender",
	// "aloe-vera",
	// "snake-plant",
	// "golden-barrel-cactus"
];

(async () => {
	const browser = await puppeteer.launch({ headless: false, defaultViewport: false });
	const page = await browser.newPage();

	await page.setViewport({ width: 1080, height: 1024 });

	for (const plant of plants) {
		// goto the page
		await page.goto(`https://unsplash.com/s/photos/${plant}`);
		// select all images
		const imageHandles = await page.$$('figure');

		for (const [index, imageHandle] of imageHandles.entries()) {
			// click on a image
			await imageHandle.click()
			const modal = await page.waitForSelector("[data-test='photos-route']")

			// download a image
			await modal.waitForSelector("#modal-portal > div > div > div > div.Lvlem.fBS9b > div > div > div:nth-child(1) > div.btXSB > div > div > button > div.omfF5 > div.MorZF > img")
			let urls = await modal.$eval("#modal-portal > div > div > div > div.Lvlem.fBS9b > div > div > div:nth-child(1) > div.btXSB > div > div > button > div.omfF5 > div.MorZF > img", elem => elem.srcset)
			urls = urls.split(",").map(url => url.trim().split(" ")[0])

			if (!urls)
				continue

			await downloadFile(urls[urls.length - 1], plant, `img-${index + 1}`)

			const closeBtn = await page.waitForSelector("#modal-portal > div > div > div > div.YcKTH > button")
			await closeBtn.click()
			await page.waitForNetworkIdle()
		}
	}

	await browser.close();
})()