# Polpostann

Scripts for tweets annotation using llm models via [vllm](https://docs.vllm.ai/en/latest) python inference library, on a HTC clusters managed with Slurm Linux job scheduler.

## Usage

1. We have one python scripts that take as arguments the llm model parameters, the sampling parameters, a csv file where the tweets to annotate are and the name of the column, as well as the system and the user prompt.     



    """
    Example of script calling using 2 gpu cards ('tensor_parallel_size' vllm model variable).
    Deepseek recommends to avoid adding a system prompt; all instructions should be contained within the user prompt.

     python tweets.py \
        --model_params='{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "guided_decoding_backend": "xgrammar", "seed": 19, "dtype": "half", "gpu_memory_utilization": 0.9, "tensor_parallel_size": 2}' \
        --sampling_params='{"temperature": 0.6, "top_p": 0.95}' \
        --tweets_file=200_sampled_xan_seed_123_fr_en.csv \
        --tweets_column=english \
        --system_prompt='' \
        --user_prompt='You are an expert in French politics. Please classify the following social media message (that were posted in the weeks leading up to the 2022 presidential election in France) according to whether it express support or positive attitudes towards Le Pen in this election. You must use only the information contained in the message. Be concise and answer only YES or NO. Here is the message: ${tweet}' \
        --guided_choice=YES,NO \
        --logfile=2Xv100DeepSeek-R1-Distill-Qwen-32B.log \
        --outfolder=2Xv100DeepSeek-R1-Distill-Qwen-32B
    """

