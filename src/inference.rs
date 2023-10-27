use std::collections::HashMap;

use crate::{data::MNISTBatcher, training::TrainingConfig};
use burn::data::dataset::source::huggingface::{MNISTDataset, MNISTItem};
use burn::data::dataset::Dataset;
use burn::record::Record;
use burn::tensor::{Element, ElementConversion, Int, Tensor};
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, items: MNISTDataset) {
    let config =
        TrainingConfig::load(format!("{artifact_dir}/config.json")).expect("A config exists");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Failed to load trained model");

    let model = config.model.init_with::<B>(record).to_device(&device);
    let batcher = MNISTBatcher::new(device);

    const ERROR_KEY: &str = "error";
    const OK_KEY: &str = "ok";
    let mut stats = HashMap::new();
    stats.insert(ERROR_KEY, 0);
    stats.insert(OK_KEY, 0);

    for i in 0..items.len() {
        let item = items.get(i).unwrap();
        let label = item.label;
        let batch = batcher.batch(vec![item]);
        let output = model.forward(batch.images);
        let predicted = output.argmax(1).flatten::<1>(0, 1);
        let label = Tensor::<B, 1, Int>::from_data([(label as i64).elem()]);

        // println!("Predicted {} Expected {}", predicted, label);
        let result = predicted.equal(label).into_data().value[0];
        if result {
            let cnt = stats.get_mut(&OK_KEY).unwrap();
            *cnt += 1;
        } else {
            let cnt = stats.get_mut(&ERROR_KEY).unwrap();
            *cnt += 1
        }


    }
    println!("ok -  {} error {}", stats.get(&OK_KEY).unwrap(), stats.get(&ERROR_KEY).unwrap());
    let accuracy = stats.get(&ERROR_KEY).unwrap() / items.len();
    println!("accuracy -  {}", accuracy);

}
