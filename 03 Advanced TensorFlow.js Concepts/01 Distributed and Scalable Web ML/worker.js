importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js');

let model;

self.onmessage = async (e) => {
    if (e.data.type === 'init_model') {
        model = tf.sequential([
            tf.layers.dense({ units: 16, activation: 'relu', inputShape: [4] }),
            tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]);
        model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });
        const X = tf.randomNormal([100, 4]);
        const y = X.sum(1).greater(tf.scalar(0)).toFloat();
        await model.fit(X, y, { epochs: 3, batchSize: 32, verbose: 0 });
        self.postMessage({ type: 'model_ready' });
    } else if (e.data.type === 'predict') {
        const input = tf.tensor2d([e.data.input], [1, 4]);
        const prediction = model.predict(input);
        const result = (await prediction.array())[0][0];
        self.postMessage({ type: 'prediction', result });
        input.dispose();
        prediction.dispose();
    }
};