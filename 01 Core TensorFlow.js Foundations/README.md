# 🏗️ Core TensorFlow.js Foundations

## 📖 Introduction
TensorFlow.js is a powerful library for building and deploying machine learning models in the browser and Node.js, leveraging JavaScript and WebGL for web-based AI. This section, **Core TensorFlow.js Foundations**, covers the essential building blocks for web ML: **Tensors and Operations**, **Models and Layers**, **Data Loading and Preprocessing**, **Training in the Browser**, and **Web Integration**. It equips learners to create basic ML models in the browser, integrate them with web applications, and prepare for AI/ML interviews, complementing your TensorFlow, Keras, Matplotlib, Pandas, and NumPy skills.

## 🎯 Learning Objectives
- Master TensorFlow.js tensors and operations for web ML computations.
- Build and configure neural network models using layers and optimizers.
- Load and preprocess browser-based data (e.g., images, webcam).
- Train models in the browser with callbacks and visualizations.
- Integrate TensorFlow.js with HTML, JavaScript, and React for web applications.

## 🔑 Key Concepts
- **Tensors and Operations**:
  - Tensor Creation (`tf.tensor`, `tf.zeros`, `tf.randomNormal`)
  - Attributes (shape, `dtype`, rank)
  - Operations (slicing, reshaping, `tf.matMul`, broadcasting)
  - WebGL Acceleration (`tf.setBackend('webgl')`)
  - JavaScript/NumPy Integration
- **Models and Layers**:
  - Defining Models (`tf.sequential`, `tf.model`)
  - Layers: Dense, Conv2D, MaxPooling2D, LSTM
  - Activations: ReLU, Sigmoid, Softmax
  - Loss Functions: MeanSquaredError, CategoricalCrossentropy
  - Optimizers: SGD, Adam
- **Data Loading and Preprocessing**:
  - Loading Data (`tf.data`, `tf.data.csv`, `tf.data.webcam`)
  - Browser-Based Datasets (MNIST, custom JSON/CSV)
  - Preprocessing (normalization, one-hot encoding)
  - Handling Images/Video (`tf.browser.fromPixels`)
  - Asynchronous Data Pipelines (`tf.data.Dataset`)
- **Training in the Browser**:
  - Training Models (`model.fit`, `model.fitDataset`)
  - Callbacks (EarlyStopping, Custom Web Logging)
  - Browser Memory Management
  - Saving/Loading Models (`model.save`, `tf.io.browserFiles`)
  - Visualization with HTML5 Canvas
- **Web Integration**:
  - Embedding Models in HTML/JavaScript
  - Real-Time Inference with WebCam/Canvas
  - Basic React Integration for ML UIs
  - Debugging Browser Performance (Chrome DevTools)

## 📝 Example Walkthroughs
The following HTML files (with embedded JavaScript) demonstrate each subsection:

1. **`tensors_operations.html`**:
   - Creates tensors (`tf.tensor`, `tf.zeros`, `tf.randomNormal`).
   - Demonstrates operations (slicing, reshaping, matrix multiplication).
   - Uses WebGL for large tensor computations.
   - Visualizes a 2D tensor on HTML5 Canvas.

   Example code:
   ```javascript
   const tensor = tf.tensor([1, 2, 3, 4], [2, 2]);
   const matMul = tensor.matMul(tensor);
   ```

2. **`models_layers.html`**:
   - Builds a sequential model with Dense layers.
   - Creates a functional model for complex architectures.
   - Trains a model on synthetic data.
   - Displays predictions for test samples.

   Example code:
   ```javascript
   const model = tf.sequential();
   model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [4] }));
   ```

3. **`data_loading.html`**:
   - Loads synthetic image data (simulating MNIST).
   - Creates a `tf.data` pipeline with shuffling, batching, and normalization.
   - Visualizes a sample image on Canvas.
   - Trains a simple model on the dataset.

   Example code:
   ```javascript
   const dataset = tf.data.zip({ xs: X, ys: y }).shuffle(100).batch(32);
   ```

4. **`training_browser.html`**:
   - Trains a model with `model.fit` and tfjs-vis callbacks.
   - Implements a custom callback for epoch logging.
   - Saves and loads a model using browser local storage.
   - Visualizes training history with tfjs-vis.

   Example code:
   ```javascript
   await model.fit(X, y, { epochs: 10, callbacks: tfvis.show.fitCallbacks(...) });
   ```

5. **`web_integration.html` and `web_integration_react.html`**:
   - Integrates TensorFlow.js with a webcam for real-time inference.
   - Visualizes webcam frames on Canvas.
   - Builds a React app for model training and prediction.
   - Monitors browser memory usage.

   Example code:
   ```javascript
   const tensor = tf.browser.fromPixels(video);
   model.predict(tensor);
   ```

## 🛠️ Practical Tasks
1. **Tensors and Operations**:
   - Create a 3D tensor and perform slicing and matrix multiplication.
   - Visualize a 2D tensor on Canvas using WebGL backend.
   - Experiment with CPU vs. WebGL performance.
2. **Models and Layers**:
   - Build a sequential model with Conv2D and Dense layers.
   - Train a model on synthetic data and predict outputs.
   - Create a functional model with multiple inputs.
3. **Data Loading and Preprocessing**:
   - Load a custom JSON dataset and preprocess with `tf.data`.
   - Normalize and visualize an image tensor on Canvas.
   - Build a data pipeline for batch training.
4. **Training in the Browser**:
   - Train a model with early stopping and tfjs-vis callbacks.
   - Save and load a model using browser storage.
   - Visualize training metrics with tfjs-vis.
5. **Web Integration**:
   - Create a webcam-based classifier with real-time predictions.
   - Build a React app to display model predictions.
   - Debug memory leaks using Chrome DevTools.

## 💡 Interview Tips
- **Common Questions**:
  - How do you create and manipulate tensors in TensorFlow.js?
  - What’s the difference between `tf.sequential` and `tf.model`?
  - How do you handle webcam data for real-time ML?
- **Tips**:
  - Explain WebGL’s role in accelerating tensor operations.
  - Highlight asynchronous data pipelines for browser efficiency.
  - Be ready to code a simple model or webcam inference.
- **Coding Tasks**:
  - Create a 2D tensor and perform matrix multiplication.
  - Build and train a sequential model in JavaScript.
  - Implement real-time webcam inference with TensorFlow.js.

## 📚 Resources
- [TensorFlow.js Documentation](https://js.tensorflow.org/)
- [TensorFlow.js Tutorials](https://www.tensorflow.org/js/tutorials)
- [tfjs-vis](https://github.com/tensorflow/tfjs-vis)
- [Learning TensorFlow.js by Gant Laborde](https://www.oreilly.com/library/view/learning-tensorflowjs/9781492090786/)
- [Google Codelabs: TensorFlow.js](https://codelabs.developers.google.com/?cat=TensorFlow)
- [MDN Web Docs: Canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)