# Galactica Language Model Experimentation

## Clonining the Code

```bash
# as standard practice I use a code directory in my home to organize repositories
cd ~
mkdir code
cd code

# create as local cache for the galactica models (refer to .devcontainer)
mkdir huggingface_cache

# clone the repository
git clone git@github.com:JohnnyFoulds/galai.git
cd galai

# launce vs code and use the dev container
code .
```

## References

## [GALPACA 30B (large)](https://huggingface.co/GeorgiaTechResearchInstitute/galpaca-30b)

The GALPACA models are trained by fine-tuning pre-trained GALACTICA models on the Alpaca dataset. GALACTICA models were trained on 106 billion tokens of open-access scientific text and data, including papers, textbooks, scientific websites, encyclopedias, and more. Fine-tuning the base GALACTICA models on the 52k instruction-response pairs in the Alpaca dataset allows users to query the GALPACA models in an instruct-response fashion.