cd Stage1
python test.py --dataroot ./datasets/data --name name --cls --hebing --chutu ./datasets/data/testC --input_nc 4 --output_nc 4 --lab --shuchu 'fake_B'
cd ../Stage2
python test.py --dataroot ../Stage1/datasets/data --name name --cls --hebing --chutu ./datasets/data/testC --shuchu 'fake_B'
