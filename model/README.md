# Model storage

Do not commit downloaded weights here. During training the base weights reside in Colab's runtime/cache. Publish the resulting LoRA adapter to Hugging Face and keep a backup. Local app users download the original base model and compatible adapter; library caches can live outside this folder. Record exact base revision and adapter revision for reproducibility.
