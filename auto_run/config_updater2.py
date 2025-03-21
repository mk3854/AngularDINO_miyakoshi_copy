import yaml
import subprocess

def update_config(config_file, new_config):
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # 新しい設定で既存の設定を更新
    data.update(new_config)

    with open(config_file, 'w') as f:
        yaml.dump(data, f)

def run_script(script_name, config_file):
    subprocess.run(['python', script_name, '--config', config_file])

# 変更したい設定のリスト
config_list = [
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial1/mnist_class1_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [1], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial1 step50 pkh3506a gpu0 PUMNIST normalclass1 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial1/mnist_class2_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [2], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial1 step50 pkh3506a gpu0 PUMNIST normalclass2 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial1/mnist_class3_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [3], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial1 step50 pkh3506a gpu0 PUMNIST normalclass3 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial1/mnist_class4_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [4], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial1 step50 pkh3506a gpu0 PUMNIST normalclass4 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial1/mnist_class5_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [5], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial1 step50 pkh3506a gpu0 PUMNIST normalclass5 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial1/mnist_class6_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [6], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial1 step50 pkh3506a gpu0 PUMNIST normalclass6 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial1/mnist_class7_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [7], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial1 step50 pkh3506a gpu0 PUMNIST normalclass7 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial1/mnist_class8_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [8], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial1 step50 pkh3506a gpu0 PUMNIST normalclass8 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial1/mnist_class9_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [9], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial1 step50 pkh3506a gpu0 PUMNIST normalclass9 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},

    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial2/mnist_class1_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [1], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial2 step50 pkh3506a gpu0 PUMNIST normalclass1 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial2/mnist_class2_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [2], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial2 step50 pkh3506a gpu0 PUMNIST normalclass2 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial2/mnist_class3_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [3], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial2 step50 pkh3506a gpu0 PUMNIST normalclass3 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial2/mnist_class4_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [4], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial2 step50 pkh3506a gpu0 PUMNIST normalclass4 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial2/mnist_class5_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [5], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial2 step50 pkh3506a gpu0 PUMNIST normalclass5 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial2/mnist_class6_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [6], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial2 step50 pkh3506a gpu0 PUMNIST normalclass6 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial2/mnist_class7_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [7], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial2 step50 pkh3506a gpu0 PUMNIST normalclass7 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial2/mnist_class8_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [8], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial2 step50 pkh3506a gpu0 PUMNIST normalclass8 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial2/mnist_class9_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [9], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial2 step50 pkh3506a gpu0 PUMNIST normalclass9 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},

    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial3/mnist_class1_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [1], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial3 step50 pkh3506a gpu0 PUMNIST normalclass1 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial3/mnist_class2_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [2], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial3 step50 pkh3506a gpu0 PUMNIST normalclass2 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial3/mnist_class3_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [3], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial3 step50 pkh3506a gpu0 PUMNIST normalclass3 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial3/mnist_class4_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [4], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial3 step50 pkh3506a gpu0 PUMNIST normalclass4 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial3/mnist_class5_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [5], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial3 step50 pkh3506a gpu0 PUMNIST normalclass5 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial3/mnist_class6_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [6], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial3 step50 pkh3506a gpu0 PUMNIST normalclass6 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial3/mnist_class7_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [7], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial3 step50 pkh3506a gpu0 PUMNIST normalclass7 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial3/mnist_class8_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [8], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial3 step50 pkh3506a gpu0 PUMNIST normalclass8 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial3/mnist_class9_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [9], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial3 step50 pkh3506a gpu0 PUMNIST normalclass9 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},

    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial4/mnist_class1_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [1], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial4 step50 pkh3506a gpu0 PUMNIST normalclass1 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial4/mnist_class2_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [2], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial4 step50 pkh3506a gpu0 PUMNIST normalclass2 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial4/mnist_class3_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [3], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial4 step50 pkh3506a gpu0 PUMNIST normalclass3 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial4/mnist_class4_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [4], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial4 step50 pkh3506a gpu0 PUMNIST normalclass4 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial4/mnist_class5_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [5], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial4 step50 pkh3506a gpu0 PUMNIST normalclass5 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial4/mnist_class6_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [6], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial4 step50 pkh3506a gpu0 PUMNIST normalclass6 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial4/mnist_class7_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [7], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial4 step50 pkh3506a gpu0 PUMNIST normalclass7 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial4/mnist_class8_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [8], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial4 step50 pkh3506a gpu0 PUMNIST normalclass8 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial4/mnist_class9_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [9], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial4 step50 pkh3506a gpu0 PUMNIST normalclass9 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},

    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial5/mnist_class1_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [1], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial5 step50 pkh3506a gpu0 PUMNIST normalclass1 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial5/mnist_class2_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [2], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial5 step50 pkh3506a gpu0 PUMNIST normalclass2 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial5/mnist_class3_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [3], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial5 step50 pkh3506a gpu0 PUMNIST normalclass3 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial5/mnist_class4_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [4], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial5 step50 pkh3506a gpu0 PUMNIST normalclass4 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial5/mnist_class5_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [5], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial5 step50 pkh3506a gpu0 PUMNIST normalclass5 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial5/mnist_class6_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [6], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial5 step50 pkh3506a gpu0 PUMNIST normalclass6 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial5/mnist_class7_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [7], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial5 step50 pkh3506a gpu0 PUMNIST normalclass7 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial5/mnist_class8_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [8], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial5 step50 pkh3506a gpu0 PUMNIST normalclass8 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},
    {'output_dir': '/workspace/angular_dino/outputs/test/mnist/trials/tea-stu-arcface-m0.3-step50/trial5/mnist_class9_resnet18_ptnum100_ep200_bs128_lambda1.0-warmup100_tea-stu-arcface-m0.3-step50-num10_useweight_smallhead_emb64_lr0.0005-weighted-sampler_test', 
     'normal_class': [9], 'margin': 0.30, 'margin_type': 'tea-stu', 
     'task_name': "tea-stu-arcface-trial5 step50 pkh3506a gpu0 PUMNIST normalclass9 resnet18 ptnum100 ep200 bs128 lambda1.0 arcface0.3 useweight smallhead embdim64 lr0.0005 weighted sampler test"},

]


# config.ymlファイル名とPythonスクリプト名
config_file = 'auto_config2.yml'
script_name = 'angular_dino_train.py'

# 各設定でスクリプトを実行
for config in config_list:
    update_config(config_file, config)
    run_script(script_name, config_file)
    # break