# Prompting LLMs for Perceived Gendered Style Stransfer

This repository contains code and data for the paper **“Contrast Comes at a Cost: Prompting Large Language Models for Perception-Centered Gendered Style Transfer.”**

## About

This project studies whether large language models can rewrite the same text in styles perceived as feminine and masculine while preserving its meaning.

We compare four prompting strategies:

- author identity;
- gendered voice;
- general gendered style;
- fine-grained style instructions.

Experiments cover Gemma 3, Gemma 2, Qwen2.5, Llama 3.1, and DeepSeek-V3 using 500 texts sampled from [PASTEL](https://aclanthology.org/D19-1179/).



## Scripts

- `rewrite-script/`: generation and evaluation scripts
- `classifier/`: perceived gendered-style classifier
- `survey/` and `survey_2/`: human-evaluation interfaces and data
- `data.zip`: 20 synthetic datasets generated across the tested models and prompting conditions

## Responsible use

The generated texts are research artifacts and should not be used for demographic profiling or treated as universal representations of how people of different genders write.

## Citation

@inproceedings{chen-etal-2026-contrast,
    title = "Contrast Comes at a Cost: Prompting Large Language Models for Perception-Centered Gendered Style Transfer",
    author = "Chen, Hongyu  and
      Falk, Neele  and
      Roth, Michael  and
      Falenska, Agnieszka",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2477/",
    pages = "56220--56258",
    ISBN = "979-8-89176-459-0",
}



## Contact

Questions: `hongyu.chen@iris.uni-stuttgart.de`
