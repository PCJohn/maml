optirun python maml_main.py --metatrain_iterations=1550 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --meta_lr=0.001 --num_updates=1 --num_classes=10 --logdir=logs/mnist10shot --num_filters=32 --target_task_lr=0.001 --target_maml_iterations=50 --target_task_iterations=1 --sgd_lr=0.001 --sgd_bz=10 --sgd_iterations=5000 --opt_mode=maml

optirun python maml_main.py --metatrain_iterations=1550 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --meta_lr=0.001 --num_updates=1 --num_classes=10 --logdir=logs/mnist10shot --num_filters=32 --target_task_lr=0.001 --target_maml_iterations=50 --target_task_iterations=1 --sgd_lr=0.001 --sgd_bz=10 --sgd_iterations=5000 --opt_mode=sgd --img_size=84
