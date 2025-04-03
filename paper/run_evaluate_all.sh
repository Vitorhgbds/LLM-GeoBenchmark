gas -t TF -c ./configs/config_llama32.toml evaluate > ./logs/evaluate/TF/llama32.txt
gas -t TF -c ./configs/config_llama31.toml evaluate > ./logs/evaluate/TF/llama31.txt
gas -t TF -c ./configs/config_ministral.toml evaluate > ./logs/evaluate/TF/ministral.txt
gas -t TF -c ./configs/config_k2.toml evaluate > ./logs/evaluate/TF/k2.txt
gas -t TF -c ./configs/config_gemma.toml evaluate > ./logs/evaluate/TF/gemma.txt
gas -t TF -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/TF/phi4.txt

gas -t TF -c ./configs/config_peft_llama32_1.toml evaluate > ./logs/evaluate/TF/peft_llama32_1.txt
gas -t TF -c ./configs/config_peft_llama31_1.toml evaluate > ./logs/evaluate/TF/peft_llama31_1.txt
gas -t TF -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v_5 evaluate > ./logs/evaluate/TF/peft_ministral_1.txt
gas -t TF -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/TF/peft_gemma_1.txt
gas -t TF -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/TF/peft_phi.txt

gas -t CHOICE -c ./configs/config_llama32.toml evaluate > ./logs/evaluate/CHOICE/llama32.txt
gas -t CHOICE -c ./configs/config_llama31.toml evaluate > ./logs/evaluate/CHOICE/llama31.txt
gas -t CHOICE -c ./configs/config_ministral.toml evaluate > ./logs/evaluate/CHOICE/ministral.txt
gas -t CHOICE -c ./configs/config_k2.toml evaluate > ./logs/evaluate/CHOICE/k2.txt
gas -t CHOICE -c ./configs/config_gemma.toml evaluate > ./logs/evaluate/CHOICE/gemma.txt
gas -t CHOICE -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/CHOICE/geogalactica.txt
gas -t CHOICE -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/CHOICE/phi4.txt

gas -t CHOICE -c ./configs/config_peft_llama32_1.toml evaluate > ./logs/evaluate/CHOICE/peft_llama32_1.txt
gas -t CHOICE -c ./configs/config_peft_llama31_1.toml evaluate > ./logs/evaluate/CHOICE/peft_llama31_1.txt
gas -t CHOICE -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/CHOICE/peft_ministral_1.txt
gas -t CHOICE -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/CHOICE/peft_gemma_1.txt
gas -t CHOICE -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/CHOICE/peft_phi.txt

gas -t COMPLETION -c ./configs/config_llama32.toml evaluate > ./logs/evaluate/COMPLETION/llama32.txt
gas -t COMPLETION -c ./configs/config_llama31.toml evaluate > ./logs/evaluate/COMPLETION/llama31.txt
gas -t COMPLETION -c ./configs/config_ministral.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_p evaluate > ./logs/evaluate/COMPLETION/ministral.txt
gas -t COMPLETION -c ./configs/config_k2.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v_5 evaluate > ./logs/evaluate/COMPLETION/k2.txt
gas -t COMPLETION -c ./configs/config_gemma.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v_5 evaluate > ./logs/evaluate/COMPLETION/gemma.txt
gas -t COMPLETION -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/COMPLETION/geogalactica.txt
gas -t COMPLETION -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/COMPLETION/phi4.txt

gas -t COMPLETION -c ./configs/config_peft_llama32_1.toml evaluate > ./logs/evaluate/COMPLETION/peft_llama32_1.txt
gas -t COMPLETION -c ./configs/config_peft_llama31_1.toml evaluate > ./logs/evaluate/COMPLETION/peft_llama31_1.txt
gas -t COMPLETION -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/COMPLETION/peft_ministral_1.txt
gas -t COMPLETION -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/COMPLETION/peft_gemma_1.txt
gas -t COMPLETION -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/COMPLETION/peft_phi.txt

gas -t QA -c ./configs/config_llama32.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/QA/llama32.txt
gas -t QA -c ./configs/config_llama31.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_p evaluate > ./logs/evaluate/QA/llama31.txt
gas -t QA -c ./configs/config_ministral.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/QA/ministral.txt
gas -t QA -c ./configs/config_k2.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_p evaluate > ./logs/evaluate/QA/k2.txt
gas -t QA -c ./configs/config_gemma.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/QA/gemma.txt
gas -t QA -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/QA/geogalactica.txt
gas -t QA -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_p evaluate > ./logs/evaluate/QA/phi4.txt

gas -t QA -c ./configs/config_peft_llama32_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_p evaluate > ./logs/evaluate/QA/peft_llama32_1.txt
gas -t QA -c ./configs/config_peft_llama31_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/QA/peft_llama31_1.txt
gas -t QA -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_p evaluate > ./logs/evaluate/QA/peft_ministral_1.txt
gas -t QA -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/QA/peft_gemma_1.txt
gas -t QA -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/QA/peft_phi.txt

gas -t NOUN -c ./configs/config_llama32.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v_5 evaluate > ./logs/evaluate/NOUN/llama32.txt
gas -t NOUN -c ./configs/config_llama31.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v_5 evaluate > ./logs/evaluate/NOUN/llama31.txt
gas -t NOUN -c ./configs/config_ministral.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/NOUN/ministral.txt
gas -t NOUN -c ./configs/config_k2.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_p evaluate > ./logs/evaluate/NOUN/k2.txt
gas -t NOUN -c ./configs/config_gemma.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/NOUN/gemma.txt
gas -t NOUN -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v evaluate > ./logs/evaluate/NOUN/geogalactica.txt
gas -t NOUN -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/NOUN/phi4.txt

gas -t NOUN -c ./configs/config_peft_llama32_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_p evaluate > ./logs/evaluate/NOUN/peft_llama32_1.txt
gas -t NOUN -c ./configs/config_peft_llama31_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/NOUN/peft_llama31_1.txt
gas -t NOUN -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_l evaluate > ./logs/evaluate/NOUN/peft_ministral_1.txt
gas -t NOUN -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_p evaluate > ./logs/evaluate/NOUN/peft_gemma_1.txt
gas -t NOUN -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_p evaluate > ./logs/evaluate/NOUN/peft_phi.txt
