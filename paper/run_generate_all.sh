# gas -l DEBUG -t TF -c ./configs/config_llama32.toml generate > ./logs/TF/llama32.txt
# gas -l DEBUG -t TF -c ./configs/config_llama31.toml generate > ./logs/TF/llama31.txt
# gas -l DEBUG -t TF -c ./configs/config_ministral.toml generate > ./logs/TF/ministral.txt
# gas -l DEBUG -t TF -c ./configs/config_k2.toml generate > ./logs/TF/k2.txt
# gas -l DEBUG -t TF -c ./configs/config_gemma.toml generate > ./logs/TF/gemma.txt
# gas -l DEBUG -t TF -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/geogalactica.txt
# gas -l DEBUG -t TF -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/phi4.txt


# gas -l DEBUG -t TF -c ./configs/config_peft_llama32_1.toml generate > ./logs/TF/peft_llama32_1.txt
# gas -l DEBUG -t TF -c ./configs/config_peft_llama31_1.toml generate > ./logs/TF/peft_llama31_1.txt
# gas -l DEBUG -t TF -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/peft_ministral_1.txt
# gas -l DEBUG -t TF -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/peft_gemma_1.txt
# gas -l DEBUG -t TF -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/peft_phi.txt
# mkdir ./logs/TF
# gas -l DEBUG -t TF -c ./configs/config_deepseek.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/deepseek.txt
# gas -l DEBUG -t TF -c ./configs/config_llama31_405b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/llama31_405b.txt
# gas -l DEBUG -t TF -c ./configs/config_llama33_70b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/llama33_70b.txt
# gas -l DEBUG -t TF -c ./configs/config_llama34_17b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/llama34_17b.txt
# gas -l DEBUG -t TF -c ./configs/config_llama34_scout.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/llama34_scout.txt
# gas -l DEBUG -t TF -c ./configs/config_ministral3_large.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/ministral3_large.txt
# gas -l DEBUG -t TF -c ./configs/config_mixtral_7b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/TF/mixtral_7b.txt

# gas -l DEBUG -t CHOICE -c ./configs/config_llama32.toml generate > ./logs/CHOICE/llama32.txt
# gas -l DEBUG -t CHOICE -c ./configs/config_llama31.toml generate > ./logs/CHOICE/llama31.txt
# gas -l DEBUG -t CHOICE -c ./configs/config_ministral.toml generate > ./logs/CHOICE/ministral.txt
# gas -l DEBUG -t CHOICE -c ./configs/config_k2.toml generate > ./logs/CHOICE/k2.txt
# gas -l DEBUG -t CHOICE -c ./configs/config_gemma.toml generate > ./logs/CHOICE/gemma.txt
# gas -l DEBUG -t CHOICE -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/CHOICE/geogalactica.txt
# gas -l DEBUG -t CHOICE -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/CHOICE/phi4.txt

# gas -l DEBUG -t CHOICE -c ./configs/config_peft_llama32_1.toml generate > ./logs/CHOICE/peft_llama32_1.txt
# gas -l DEBUG -t CHOICE -c ./configs/config_peft_llama31_1.toml generate > ./logs/CHOICE/peft_llama31_1.txt
# gas -l DEBUG -t CHOICE -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/CHOICE/peft_ministral_1.txt
# gas -l DEBUG -t CHOICE -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/CHOICE/peft_gemma_1.txt
# gas -l DEBUG -t CHOICE -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/CHOICE/peft_phi.txt
mkdir ./logs/CHOICE
gas -l DEBUG -t CHOICE -c ./configs/config_deepseek.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >        ./logs/CHOICE/deepseek.txt
gas -l DEBUG -t CHOICE -c ./configs/config_llama31_405b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >    ./logs/CHOICE/llama31_405b.txt
gas -l DEBUG -t CHOICE -c ./configs/config_llama33_70b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >     ./logs/CHOICE/llama33_70b.txt
gas -l DEBUG -t CHOICE -c ./configs/config_llama34_17b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >    ./logs/CHOICE/llama34_17b.txt
gas -l DEBUG -t CHOICE -c ./configs/config_llama34_scout.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >  ./logs/CHOICE/llama34_scout.txt
#gas -l DEBUG -t CHOICE -c ./configs/config_ministral3_large.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >./logs/CHOICE/ministral3_large.txt
gas -l DEBUG -t CHOICE -c ./configs/config_mixtral_7b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >      ./logs/CHOICE/mixtral_7b.txt

# # gas -l DEBUG -t COMPLETION -c ./configs/config_llama32.toml generate > ./logs/COMPLETION/llama32.txt
# # gas -l DEBUG -t COMPLETION -c ./configs/config_llama31.toml generate > ./logs/COMPLETION/llama31.txt
# # gas -l DEBUG -t COMPLETION -c ./configs/config_ministral.toml generate > ./logs/COMPLETION/ministral.txt
# # gas -l DEBUG -t COMPLETION -c ./configs/config_k2.toml generate > ./logs/COMPLETION/k2.txt
# # gas -l DEBUG -t COMPLETION -c ./configs/config_gemma.toml generate > ./logs/COMPLETION/gemma.txt
# # gas -l DEBUG -t COMPLETION -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/COMPLETION/phi4.txt

# # gas -l DEBUG -t COMPLETION -c ./configs/config_peft_llama32_1.toml generate > ./logs/COMPLETION/peft_llama32_1.txt
# # gas -l DEBUG -t COMPLETION -c ./configs/config_peft_llama31_1.toml generate > ./logs/COMPLETION/peft_llama31_1.txt
# # gas -l DEBUG -t COMPLETION -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/COMPLETION/peft_ministral_1.txt
# # gas -l DEBUG -t COMPLETION -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/COMPLETION/peft_gemma_1.txt
# # gas -l DEBUG -t COMPLETION -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/COMPLETION/peft_phi.txt
mkdir ./logs/COMPLETION
gas -l DEBUG -t COMPLETION -c ./configs/config_deepseek.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >        ./logs/COMPLETION/deepseek.txt
gas -l DEBUG -t COMPLETION -c ./configs/config_llama31_405b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >    ./logs/COMPLETION/llama31_405b.txt
gas -l DEBUG -t COMPLETION -c ./configs/config_llama33_70b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >     ./logs/COMPLETION/llama33_70b.txt
gas -l DEBUG -t COMPLETION -c ./configs/config_llama34_17b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >    ./logs/COMPLETION/llama34_17b.txt
gas -l DEBUG -t COMPLETION -c ./configs/config_llama34_scout.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >  ./logs/COMPLETION/llama34_scout.txt
#gas -l DEBUG -t COMPLETION -c ./configs/config_ministral3_large.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >./logs/COMPLETION/ministral3_large.txt
gas -l DEBUG -t COMPLETION -c ./configs/config_mixtral_7b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >      ./logs/COMPLETION/mixtral_7b.txt

# # gas -l DEBUG -t QA -c ./configs/config_llama32.toml generate > ./logs/QA/llama32.txt
# # gas -l DEBUG -t QA -c ./configs/config_llama31.toml generate > ./logs/QA/llama31.txt
# # gas -l DEBUG -t QA -c ./configs/config_ministral.toml generate > ./logs/QA/ministral.txt
# # gas -l DEBUG -t QA -c ./configs/config_k2.toml generate > ./logs/QA/k2.txt
# # gas -l DEBUG -t QA -c ./configs/config_gemma.toml generate > ./logs/QA/gemma.txt
# # gas -l DEBUG -t QA -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/QA/geogalactica.txt
# # gas -l DEBUG -t QA -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/QA/phi4.txt

# # gas -l DEBUG -t QA -c ./configs/config_peft_llama32_1.toml generate > ./logs/QA/peft_llama32_1.txt
# # gas -l DEBUG -t QA -c ./configs/config_peft_llama31_1.toml generate > ./logs/QA/peft_llama31_1.txt
# # gas -l DEBUG -t QA -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/QA/peft_ministral_1.txt
# # gas -l DEBUG -t QA -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/QA/peft_gemma_1.txt
# # gas -l DEBUG -t QA -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/QA/peft_phi.txt
mkdir ./logs/QA
gas -l DEBUG -t QA -c ./configs/config_deepseek.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >        ./logs/QA/deepseek.txt
gas -l DEBUG -t QA -c ./configs/config_llama31_405b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >    ./logs/QA/llama31_405b.txt
gas -l DEBUG -t QA -c ./configs/config_llama33_70b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >     ./logs/QA/llama33_70b.txt
gas -l DEBUG -t QA -c ./configs/config_llama34_17b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >    ./logs/QA/llama34_17b.txt
gas -l DEBUG -t QA -c ./configs/config_llama34_scout.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >  ./logs/QA/llama34_scout.txt
#gas -l DEBUG -t QA -c ./configs/config_ministral3_large.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >./logs/QA/ministral3_large.txt
gas -l DEBUG -t QA -c ./configs/config_mixtral_7b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >      ./logs/QA/mixtral_7b.txt

# # gas -l DEBUG -t NOUN -c ./configs/config_llama32.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/llama32.txt
# # gas -l DEBUG -t NOUN -c ./configs/config_llama31.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/llama31.txt
# # gas -l DEBUG -t NOUN -c ./configs/config_ministral.toml generate > ./logs/NOUN/ministral.txt
# # gas -l DEBUG -t NOUN -c ./configs/config_k2.toml generate > ./logs/NOUN/k2.txt
# # gas -l DEBUG -t NOUN -c ./configs/config_gemma.toml generate > ./logs/NOUN/gemma.txt
# # gas -l DEBUG -t NOUN -c ./configs/config_geogalactica.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/geogalactica.txt
# # gas -l DEBUG -t NOUN -c ./configs/config_phi4.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/phi4.txt

# # gas -l DEBUG -t NOUN -c ./configs/config_peft_llama32_1.toml generate > ./logs/NOUN/peft_llama32_1.txt
# # gas -l DEBUG -t NOUN -c ./configs/config_peft_llama31_1.toml generate > ./logs/NOUN/peft_llama31_1.txt
# # gas -l DEBUG -t NOUN -c ./configs/config_peft_ministral_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/peft_ministral_1.txt
# # gas -l DEBUG -t NOUN -c ./configs/config_peft_gemma_1.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/peft_gemma_1.txt
# # gas -l DEBUG -t NOUN -c ./configs/config_peft_phi.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate > ./logs/NOUN/peft_phi.txt
mkdir ./logs/NOUN
gas -l DEBUG -t NOUN -c ./configs/config_deepseek.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >        ./logs/NOUN/deepseek.txt
gas -l DEBUG -t NOUN -c ./configs/config_llama31_405b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >    ./logs/NOUN/llama31_405b.txt
gas -l DEBUG -t NOUN -c ./configs/config_llama33_70b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >     ./logs/NOUN/llama33_70b.txt
gas -l DEBUG -t NOUN -c ./configs/config_llama34_17b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >    ./logs/NOUN/llama34_17b.txt
gas -l DEBUG -t NOUN -c ./configs/config_llama34_scout.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >  ./logs/NOUN/llama34_scout.txt
#gas -l DEBUG -t NOUN -c ./configs/config_ministral3_large.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >./logs/NOUN/ministral3_large.txt
gas -l DEBUG -t NOUN -c ./configs/config_mixtral_7b.toml -e /mnt/D-SSD/LLM-GeoBenchmark/env/.env_v generate >      ./logs/NOUN/mixtral_7b.txt