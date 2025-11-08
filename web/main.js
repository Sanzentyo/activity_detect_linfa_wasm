import init, { predict_activity, load_model } from "../pkg/linfa_playground.js";

// Utility: generate an array of 6 float32 values in a reasonable range
function randomFeatures() {
    // mean_x, mean_y, mean_z in [-1.5, 1.5], std_x, std_y, std_z in [0, 2]
    const means = Array.from({ length: 3 }, () => (Math.random() * 3 - 1.5));
    const stds = Array.from({ length: 3 }, () => (Math.random() * 2));
    return new Float32Array([...means, ...stds]);
}

function renderResult(root, features, pred, elapsed) {
    const feat = Array.from(features).map(v => v.toFixed(4));
    root.innerHTML = `
    <table>
      <thead>
        <tr><th>feature</th><th>value</th></tr>
      </thead>
      <tbody>
        <tr><td>mean_x</td><td>${feat[0]}</td></tr>
        <tr><td>mean_y</td><td>${feat[1]}</td></tr>
        <tr><td>mean_z</td><td>${feat[2]}</td></tr>
        <tr><td>std_x</td><td>${feat[3]}</td></tr>
        <tr><td>std_y</td><td>${feat[4]}</td></tr>
        <tr><td>std_z</td><td>${feat[5]}</td></tr>
      </tbody>
    </table>
    <p>予測クラス (数値ラベル): <strong>${pred}</strong></p>
    <p>予測にかかった時間: ${elapsed} ms</p>
  `;
}

async function main() {
    // Initialize the generated wasm JS glue
    await init();

    const loadTime = document.getElementById("load-time");
    const loadBtn = document.getElementById("load-model");

    loadBtn.addEventListener("click", () => {
        const start = performance.now();
        load_model();
        const end = performance.now();
        const duration = end - start;
        loadTime.textContent = `モデルの読み込みにかかった時間: ${duration.toFixed(2)} ms`;
        console.log(`Model loading took ${duration} ms`);
    });

    const out = document.getElementById("out");
    const runBtn = document.getElementById("run");

    runBtn.addEventListener("click", () => {
        const start = performance.now();
        const f = randomFeatures();
        // wasm-bindgen can accept a regular array or TypedArray for Vec<f32>
        const pred = predict_activity(f);
        const end = performance.now();
        const elapsed = end - start;
        console.log(`Prediction took ${elapsed} ms`);
        renderResult(out, f, pred, elapsed);
    });

    // Run once on load
    //runBtn.click();
}

main().catch(err => {
    console.error(err);
    document.getElementById("out").textContent = "初期化に失敗しました: " + err;
});
