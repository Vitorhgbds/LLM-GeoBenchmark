gas -t TF -c ./configs/config_llama32.toml generate > ./logs/TF/llama32.txt
gas -t TF -c ./configs/config_llama31.toml generate > ./logs/TF/llama31.txt
gas -t TF -c ./configs/config_ministral.toml generate > ./logs/TF/ministral.txt
gas -t TF -c ./configs/config_k2.toml generate > ./logs/TF/k2.txt
gas -t TF -c ./configs/config_gemma.toml generate > ./logs/TF/gemma.txt
gas -t TF -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/geogalactica.txt
gas -t TF -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/phi4.txt

gas -t TF -c ./configs/config_peft_llama32_1.toml generate > ./logs/TF/peft_llama32_1.txt
gas -t TF -c ./configs/config_peft_llama31_1.toml generate > ./logs/TF/peft_llama31_1.txt
gas -t TF -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/peft_ministral_1.txt
gas -t TF -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/peft_gemma_1.txt
gas -t TF -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/peft_phi.txt

gas -t CHOICE -c ./configs/config_llama32.toml generate > ./logs/CHOICE/llama32.txt
gas -t CHOICE -c ./configs/config_llama31.toml generate > ./logs/CHOICE/llama31.txt
gas -t CHOICE -c ./configs/config_ministral.toml generate > ./logs/CHOICE/ministral.txt
gas -t CHOICE -c ./configs/config_k2.toml generate > ./logs/CHOICE/k2.txt
gas -t CHOICE -c ./configs/config_gemma.toml generate > ./logs/CHOICE/gemma.txt
gas -t CHOICE -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/CHOICE/geogalactica.txt
gas -t CHOICE -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/CHOICE/phi4.txt

gas -t CHOICE -c ./configs/config_peft_llama32_1.toml generate > ./logs/CHOICE/peft_llama32_1.txt
gas -t CHOICE -c ./configs/config_peft_llama31_1.toml generate > ./logs/CHOICE/peft_llama31_1.txt
gas -t CHOICE -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/CHOICE/peft_ministral_1.txt
gas -t CHOICE -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/CHOICE/peft_gemma_1.txt
gas -t CHOICE -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/CHOICE/peft_phi.txt

gas -t COMPLETION -c ./configs/config_llama32.toml generate > ./logs/COMPLETION/llama32.txt
gas -t COMPLETION -c ./configs/config_llama31.toml generate > ./logs/COMPLETION/llama31.txt
gas -t COMPLETION -c ./configs/config_ministral.toml generate > ./logs/COMPLETION/ministral.txt
gas -t COMPLETION -c ./configs/config_k2.toml generate > ./logs/COMPLETION/k2.txt
gas -t COMPLETION -c ./configs/config_gemma.toml generate > ./logs/COMPLETION/gemma.txt
gas -t COMPLETION -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/COMPLETION/phi4.txt

gas -t COMPLETION -c ./configs/config_peft_llama32_1.toml generate > ./logs/COMPLETION/peft_llama32_1.txt
gas -t COMPLETION -c ./configs/config_peft_llama31_1.toml generate > ./logs/COMPLETION/peft_llama31_1.txt
gas -t COMPLETION -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/COMPLETION/peft_ministral_1.txt
gas -t COMPLETION -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/COMPLETION/peft_gemma_1.txt
gas -t COMPLETION -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/COMPLETION/peft_phi.txt


gas -t QA -c ./configs/config_llama32.toml generate > ./logs/QA/llama32.txt
gas -t QA -c ./configs/config_llama31.toml generate > ./logs/QA/llama31.txt
gas -t QA -c ./configs/config_ministral.toml generate > ./logs/QA/ministral.txt
gas -t QA -c ./configs/config_k2.toml generate > ./logs/QA/k2.txt
gas -t QA -c ./configs/config_gemma.toml generate > ./logs/QA/gemma.txt
gas -t QA -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/QA/geogalactica.txt
gas -t QA -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/QA/phi4.txt

gas -t QA -c ./configs/config_peft_llama32_1.toml generate > ./logs/QA/peft_llama32_1.txt
gas -t QA -c ./configs/config_peft_llama31_1.toml generate > ./logs/QA/peft_llama31_1.txt
gas -t QA -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/QA/peft_ministral_1.txt
gas -t QA -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/QA/peft_gemma_1.txt
gas -t QA -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/QA/peft_phi.txt

gas -t NOUN -c ./configs/config_llama32.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/llama32.txt
gas -t NOUN -c ./configs/config_llama31.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/llama31.txt
gas -t NOUN -c ./configs/config_ministral.toml generate > ./logs/NOUN/ministral.txt
gas -t NOUN -c ./configs/config_k2.toml generate > ./logs/NOUN/k2.txt
gas -t NOUN -c ./configs/config_gemma.toml generate > ./logs/NOUN/gemma.txt
gas -t NOUN -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/geogalactica.txt
gas -t NOUN -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/phi4.txt

gas -t NOUN -c ./configs/config_peft_llama32_1.toml generate > ./logs/NOUN/peft_llama32_1.txt
gas -t NOUN -c ./configs/config_peft_llama31_1.toml generate > ./logs/NOUN/peft_llama31_1.txt
gas -t NOUN -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/peft_ministral_1.txt
gas -t NOUN -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/peft_gemma_1.txt
gas -t NOUN -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/peft_phi.txt
