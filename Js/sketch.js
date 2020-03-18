let neural;
let training
let testing
let Tdata, Tlabel;
let starting = true;

function preload() {
	neural = new neural_net(2, [
		[3, "sigmoid"],
	], 4, {
		useMomentum: true
	});
}

function setup() {
	createCanvas(100, 100);
	background(0);
	collectData();
}

function draw() {
	if (frameCount % 1 == 0 && starting) train(100);
}

function train(x = 10) {
	// for (let i = 0; i < x; i++) {
	// 	let r = random(training);
	// 	neural.train([r.x, r.y], r.label)

	// }
	neural.train(Tdata, Tlabel);
	show()
	console.log(neural.lr);


}

function show() {
	let img = get();
	img.loadPixels();
	for (let index = 0; index < height; index++) {
		for (let i1 = 0; i1 < width; i1++) {
			let r = neural.feedforward([i1 / width, index / height]);
			let rr = r.indexOf(Math.max(...r));
			switch (rr) {
				case 0:
					img.pixels[index * height * 4 + i1 * 4] = 0
					img.pixels[index * height * 4 + i1 * 4 + 1] = 0
					img.pixels[index * height * 4 + i1 * 4 + 2] = 255
					break;
				case 1:
					img.pixels[index * height * 4 + i1 * 4] = 255
					img.pixels[index * height * 4 + i1 * 4 + 1] = 0
					img.pixels[index * height * 4 + i1 * 4 + 2] = 0
					break;
				case 2:
					img.pixels[index * height * 4 + i1 * 4] = 0
					img.pixels[index * height * 4 + i1 * 4 + 1] = 255
					img.pixels[index * height * 4 + i1 * 4 + 2] = 0
					break;
				case 3:
					img.pixels[index * height * 4 + i1 * 4] = 255
					img.pixels[index * height * 4 + i1 * 4 + 1] = 255
					img.pixels[index * height * 4 + i1 * 4 + 2] = 0
					break;
				default:
					console.log("not existed");
					return;
			}
		}
	}
	img.updatePixels();
	image(img, 0, 0, width, height);

}

function mouseClicked() {
	if (mouseX > width || mouseY > height || mouseX < 0 || mouseY < 0) return;
	let r = neural.feedforward([mouseX / width, mouseY / height]);

	let rr = r.indexOf(Math.max(...r));
	switch (rr) {
		case 0:
			console.log("blue" + " with confidence of " + Math.round(r[rr] * 100) + "%");
			break;
		case 1:
			console.log("red" + " with confidence of " + Math.round(r[rr] * 100) + "%");
			break;
		case 2:
			console.log("green" + " with confidence of " + Math.round(r[rr] * 100) + "%");
			break;
		case 3:
			console.log("yellow" + " with confidence of " + Math.round(r[rr] * 100) + "%");
			break;
		default:
			console.log("not existed" + " with confidence of " + Math.round(r[rr] * 100) + "%");
			return;
	}
}

function collectData() {
	let data = []
	for (let i = 0; i < width; i++) {
		for (let i1 = 0; i1 < height; i1++) {
			if (i < width / 2) {
				if (i1 < height / 2) {
					data.push({
						x: i / width,
						y: i1 / height,
						label: [1, 0, 0, 0]
					})
				} else {
					data.push({
						x: i / width,
						y: i1 / height,
						label: [0, 0, 1, 0]
					})
				}
			} else {
				if (i1 < height / 2) {
					data.push({
						x: i / width,
						y: i1 / height,
						label: [0, 1, 0, 0]
					})
				} else {
					data.push({
						x: i / width,
						y: i1 / height,
						label: [0, 0, 0, 1]
					})
				}
			}
		}

	}
	shuffle(data, true);
	training = data;
	Tdata = data.map((d) => {
		return [d.x, d.y]
	})
	Tlabel = data.map((d) => {
		return d.label
	})
}