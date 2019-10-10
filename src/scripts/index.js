var LinearLearner = {

	init: function () {

		var

			/**
			 * Location size displayed on canvas
			 */
			dotSize

			/**
			 * Color of the location displayed on canvas
			 */
			, dotColor

			/**
			 * Minimum interval value
			 */
			, minVal

			/**
			 * Maximum interval value
			 */
			, maxVal

			/**
			 * How many times to train on the dataset
			 */
			, epochs

			, xTrainData

			, yTrainData

			/**
			 * 2d array of points [[x1,y1], [x2,y2]]
			 */
			, coordinates = []

			, canvas

			, ctx

			, xInputElement

			/**
			 * Returns the matching element by id
			 */
			, getElementById = function (id) {
				return document.getElementById(id);
			}

			/**
			 * Draws a rectangle on the canvas at the specified coordinate
			 * @param { object } ctx canvas 2d context
			 * @param { number } x x coordinate
			 * @param { number } y y coordinate
			 * @param { string } color the color in which the rectangle will be drawn
			 * @param { number } size the size of the square
			 */
			, createRectangle = function (ctx, x, y, color = dotColor, size = dotSize) {
				ctx.fillStyle = color;
				ctx.fillRect(x, y, size, size);
			}

			/**
			 * Clears the canvas
			 * @param { object } ctx canvas 2d context
			 * @param { number } width width of the canvas
			 * @param { number } height height of the canvas
			 */
			, clearCanvas = function (ctx, width, height) {
				ctx.clearRect(0, 0, width, height);
			}

			/**
			 * Creates a gradient of a specific size and assigns it to the canvas context
			 * @param { object } ctx canvas 2d context
			 * @param { number } size width and height of the gradient
			 */
			, createGradient = function (ctx, size) {
				let gradient = ctx.createLinearGradient(0, 0, size, size);
				gradient.addColorStop('0', 'magenta');
				gradient.addColorStop('0.5', 'cyan');
				gradient.addColorStop('1.0', 'red');
				ctx.strokeStyle = gradient;
			}

			/**
			 * Draws a line between two sets of coordinates
			 * @param { object } ctx canvas 2D context
			 * @param { array } start 1D numeric array
			 * @param { array } end 1D numeric array
			 */
			, drawLine = function (ctx, start, end) {
				ctx.beginPath();
				ctx.moveTo(start[0], start[1]);
				ctx.lineTo(end[0], end[1]);
				ctx.stroke();
			}

			/**
			 * Draws a circle on the canvas
			 * @param { object } ctx canvas 2D context
			 * @param { number } x x coordinate
			 * @param { number } y y coordinate
			 */
			, drawCircle = function (ctx, x, y) {
				ctx.beginPath();
				ctx.arc(x, y, 2, 0, 2 * Math.PI);
				ctx.stroke();
			}

			/**
			 * Draws a path between an array of points
			 * @param { object } ctx canvas 2D context
			 * @param { Array } pointsArray 1D numeric array
			 */
			, drawPath = function (ctx, pointsArray) {
				let destinations = pointsArray.length - 1;

				for (let i = 0; i < destinations; i++) {
					let start = pointsArray[i];
					let end = pointsArray[i + 1];
					drawLine(ctx, start, end);
				}
			}

			/**
			* Draws points on a canvas
			* @param { object } ctx canvas 2D context
			* @param { Array } arr 1D numeric array
			*/
			, drawCoordinates = function (ctx, arr) {
				let destinations = arr.length;
				ctx.fillStyle = dotColor;
				for (let i = 0; i < destinations; i++) {
					let x = arr[i][0];
					let y = arr[i][1];
					createRectangle(ctx, x, y);
				}
			}

			/**
			 * Calculates the distance between two sets of points
			 * @param { number } x1 x coordinate of the first point
			 * @param { number } y1 y coordinate of the first point
			 * @param { number } x2 x coordinate of the second point
			 * @param { number } y2 y coordinate of the second point
			 */
			, computeDistanceBetweenTwoPoints = function (x1, y1, x2, y2) {
				return Math.sqrt((Math.pow(x2 - x1, 2)) + (Math.pow(y2 - y1, 2)))
			}

			/**
			 * Returns a random generated number
			 * @param { number } min minimum value
			 * @param { number } max maximum value
			 * @returns { number } generated number
			 */
			, generateRandomNumber = function (min = minVal, max = maxVal) {
				return Math.floor(Math.random() * (max - min + 1)) + min;
			}

			/**
			 * Returns a random 2D array of maximum 100 elements
			 * @returns { array } generated array
			 */
			, generateRandom2DArray = function () {
				let generatedArray = [];

				let arraySize = generateRandomNumber(1, 100);

				for (let i = 0; i < arraySize; i++) {
					let x = generateRandomNumber();
					let y = generateRandomNumber();
					generatedArray.push([x, y]);
				}

				return generatedArray;
			}

			/**
			 * Swaps two elements in an array
			 * @param { Array } arr 1D numeric array
			 * @param { number } indexA 1D numeric array
			 * @param { number } indexB 1D numeric array
			 */
			, swapArrayElements = function (arr, indexA, indexB) {
				let temp = arr[indexA];
				arr[indexA] = arr[indexB];
				arr[indexB] = temp;
			}

			/**
			* async because it takes time to learn
			*/
			, learnLinear = async function (xInput) {
				console.log('xInput', xInput);
				// sequential model - the outputs of one layer are the inputs to the next layer
				const model = tf.sequential();

				// dense layer means that all of the nodes in each of the layers are connected to eachother
				model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

				// meanSquaredError - standard function for linear equations
				// sgd - stochastic gradient descent (how it 'learns')
				model.compile({
					loss: 'meanSquaredError',
					optimizer: 'sgd'
				});

				// we consider x values to be inputs and y to be outputs
				// so that in the future, when we input an x, we will get an y back
				// we create 2 tensors for this approach, one for x and one for y
				const xPoints = [-1, 0, 1, 2, 3, 4];
				const yPoints = [-3, -1, 1, 3, 5, 7];

				const xs = tf.tensor2d(xPoints, [6, 1]);
				const ys = tf.tensor2d(yPoints, [6, 1]);

				// now we train... and wait
				// epochs(aka iterations) is how many times we train
				await model.fit(xs, ys, { epochs: epochs });

				// predictedY = what we predict for xInput 
				const predictedYTensor = model.predict(tf.tensor2d([xInput], [1, 1]));

				// make the data readable😡
				return predictedYTensor.data();
			}

			/**
			 * Sets event listeners
			 */
			, registerEventListeners = function () {
				getElementById('btnCompute').addEventListener('click', compute);

				getElementById('btnGenerate').addEventListener('click', () => {
					xInputElement.value = JSON.stringify(generateRandom2DArray());
				});
			}

			/**
			 * Gets the canvas DOM reference and sets it's size
			 * @param {string} name the id of the canvas
			 * @param {number} width the width of the canvas
			 * @param {number} height the height of the canvas
			 */
			, getAndInitializeCanvas = function (name, width, height) {
				let canvas = getElementById(name);

				canvas.width = width;
				canvas.height = height;

				return canvas;
			}

			/**
			* Returns the 2d context of a canvas
			* @param { object } canvas
			*/
			, getCanvas2DContext = function (canvas) {
				return canvas.getContext('2d');
			}

			/**
			 * Displays stats to the user
			 * @param {Array} pathArray the optimized path
			 * @param {number} predictedY the predictedY point
			 * @param {number} elapsedTime total time elapsed
			 */
			, displayStats = function (pathArray, predictedY, elapsedTime) {
				console.log('optimizedDistance', predictedY);
				console.log('finalPath', pathArray);
				console.log(elapsedTime + ' ms');

				getElementById('predictedY').innerHTML = predictedY;
				getElementById('elapsedTime').innerHTML = elapsedTime + ' ms';
			}

			, compute = function () {
				let xInput = parseInt(xInputElement.value);

				if (!xInput) {
					return;
				}

				// Note: The browser will round this time.... thanks Spectre!👻
				ini = performance.now();

				clearCanvas(ctx, mapSize, mapSize);

				learnLinear(xInput).then(predictedYData => {
					let predictedY = predictedYData[0];

					coordinates.push([xInput, predictedY]);

					drawCoordinates(ctx, coordinates);
					drawPath(ctx, coordinates);
	
					end = performance.now();
	
					displayStats(coordinates, predictedY, (end - ini));
				});
			}

			/**
			 * Sets some initial settings
			 */
			, initializeSettings = function () {
				dotSize = 2;
				mapSize = 10;
				minVal = 0;
				maxVal = mapSize;
				dotColor = 'white';

				epochs = 250;

				xTrainData = [-1, 0, 1, 2, 3, 4];
				yTrainData = [-3, -1, 1, 3, 5, 7];

				for (let i = 0; i < xTrainData.length; i++) {
					coordinates.push([xTrainData[i], yTrainData[i]]);
				}

				canvas = getAndInitializeCanvas('foreground', mapSize, mapSize);
				ctx = getCanvas2DContext(canvas);
				createGradient(ctx, mapSize);

				xInputElement = getElementById('coordinates');
			}

			/**
			* Initializes and starts the application
			*/
			, initialize = function () {

				initializeSettings();

				registerEventListeners();
			}

		return {
			init: initialize()
		}
	}
}

LinearLearner.init();


