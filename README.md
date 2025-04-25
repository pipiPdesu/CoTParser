model and script: reference to link.txt

dataset: reference to [link](https://huggingface.co/datasets/shuyi-zsy/LLMSR/tree/main) 

To run the code,  using the command:

python final_pipeline.py --input_file test.json --model_path_perfix /root/models/


--input_file indicates the test file, and --model_path_perfix indicates where the model is.  If the model and the script file are in the same dictionary, this can be ""

It need 64G VRAM as it run four models serially.
