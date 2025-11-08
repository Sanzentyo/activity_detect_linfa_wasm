use linfa::prelude::*;
use linfa_playground::{AccelData, BincodeDecisionTree};
use linfa_trees::DecisionTree;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// Window sizeはこれを使う
use linfa_playground::WINDOW_SIZE;
use std::io::Write;
use std::path::Path;



fn main() -> anyhow::Result<()> {
    // activity -> label mapping
    let mapping = vec![
        ("sit.csv", 0usize),
        ("walk-with-hand.csv", 1usize),
        ("walking-in-pocket.csv", 2usize),
        ("climb-up.csv", 3usize),
        ("four-legged-walking.csv", 4usize),
    ];

    let mut features: Vec<[f32; 6]> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    let data_dir = Path::new("data");
    for (file_name, label) in mapping.iter() {
        let path = data_dir.join(file_name);
        if !path.exists() {
            eprintln!("warning: {} not found, skipping", path.display());
            continue;
        }

        let file = std::fs::File::open(&path)?;
        let mut csv_reader = csv::Reader::from_reader(file);
        let csv_data = csv_reader
            .deserialize::<AccelData>()
            .into_iter()
            .collect::<Result<Vec<AccelData>, csv::Error>>()?;

        // non-overlapping windows
        let mut i = 0usize;
        while i + WINDOW_SIZE <= csv_data.len() {
            let window = &csv_data[i..i + WINDOW_SIZE];
            let feat = linfa_playground::extract_window_features(window);
            features.push(feat);
            labels.push(*label);
            i += WINDOW_SIZE / 2; // stride == WINDOW_SIZE / 2
        }
    }

    if features.is_empty() {
        anyhow::bail!("no feature windows extracted — check data files");
    }

    let n_samples = features.len();
    let mut feature_array = Array2::<f32>::zeros((n_samples, 6));
    for (i, f) in features.iter().enumerate() {
        for j in 0..6 {
            feature_array[[i, j]] = f[j];
        }
    }

    let label_array = ndarray::Array1::from(labels.clone());

    let dataset = linfa::Dataset::new(feature_array, label_array);

    let mut rng = SmallRng::seed_from_u64(42);
    let (train, test) = dataset.shuffle(&mut rng).split_with_ratio(0.8);

    // Train a Decision Tree classifier (default params)
    let model = DecisionTree::params().fit(&train)?;

    let pred = model.predict(&test);

    // compute accuracy manually
    let y_true = test.targets();
    let correct = pred
        .iter()
        .zip(y_true.iter())
        .filter(|(p, t)| p == t)
        .count();
    let acc = correct as f32 / y_true.len() as f32;
    println!(
        "Test accuracy: {:.2}% ({} samples)",
        acc * 100.0,
        test.nsamples()
    );

    let mut tikz = std::fs::File::create("activity_decision_tree.tex")?;
    tikz.write_all(model.export_to_tikz().with_legend().to_string().as_bytes())?;

    // parameterの保存
    let model_path = Path::new("activity_decision_tree.bincode");
    let mut model_file = std::fs::File::create(model_path)?;
    let model = BincodeDecisionTree { tree: model };
    bincode::encode_into_std_write(&model, &mut model_file, bincode::config::standard())?;

    Ok(())
}
