python adapt_multi.py --dset HAM10000 --t 0 --max_epoch 10 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt > HAM10000_0.txt
python adapt_multi.py --dset HAM10000 --t 1 --max_epoch 10 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt > HAM10000_1.txt
python adapt_multi.py --dset HAM10000 --t 2 --max_epoch 10 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt > HAM10000_2.txt
python adapt_multi.py --dset HAM10000 --t 3 --max_epoch 10 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt > HAM10000_3.txt
python adapt_multi.py --dset DR --t 0 --max_epoch 15 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt > DR_0.txt
python adapt_multi.py --dset DR --t 1 --max_epoch 15 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt > DR_1.txt
python adapt_multi.py --dset DR --t 2 --max_epoch 15 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt > DR_2.txt