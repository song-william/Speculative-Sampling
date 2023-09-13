python generate.py  --method speculative \
                    --prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
                    --max_new_tokens 64 \
                    --draft_model /home/wsong/llama-2-7b \
                    --target_model /home/wsong/llama-2-70b \
                    --temperature 0.5
python generate.py  --method autoregressive \
                    --prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
                    --max_new_tokens 64 \
                    --draft_model /home/wsong/llama-2-7b \
                    --target_model /home/wsong/llama-2-7b \
                    --temperature 0.5