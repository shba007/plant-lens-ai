import fs from 'fs';
import axios from 'axios';

export async function downloadFile(url, plantName, fileName) {
	const fileWorkingPath = `../data/${plantName}`
	const filePath = `${fileWorkingPath}/${fileName}.jpg`

	try {
		if (!fs.existsSync(fileWorkingPath))
			fs.mkdirSync(fileWorkingPath, { recursive: true });

		const response = await axios({
			url,
			responseType: "stream"
		});
		return new Promise((resolve, reject) => {
			response.data
				.pipe(fs.createWriteStream(filePath))
				.on("finish", () => {
					console.log(`${fileName} download complete`);
					resolve();
				})
				.on("error", e => reject(e));
		});
	} catch (e) {
		console.log(e);
	}
}