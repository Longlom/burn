mod data;
mod model;
mod training;
mod inference;

use burn::autodiff::ADBackendDecorator;
use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::WgpuBackend;
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use model::ModelConfig;
use training::TrainingConfig;

use crate::inference::infer;

fn main() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;
    
    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./tmp/guide";
    // training::train::<MyAutodiffBackend>(
    //     artifact_dir,
    //     TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
    //     device.clone(),
    // );
    infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::source::huggingface::MNISTDataset::test(),
    );
}
