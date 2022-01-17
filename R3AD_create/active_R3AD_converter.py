import mmcv
import os

from R3AD_create.active_R3AD_data_utils import R3ADData

# create testset of home1 and home2

def create_indoor_info_file(data_path,
                            pkl_prefix='R3AD',
                            save_path=None,
                            use_v1=False,
                            workers=1):

    assert os.path.exists(data_path)
    assert pkl_prefix in ['R3AD']
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    train_filename = os.path.join(save_path, f'active_{pkl_prefix}_infos_train.pkl')
    val_filename = os.path.join(save_path, f'active_{pkl_prefix}_infos_val.pkl')
    train_dataset = R3ADData(
        root_path=data_path, split='train', use_v1=use_v1)
    val_dataset = R3ADData(
        root_path=data_path, split='test', use_v1=use_v1)


    infos_train = train_dataset.get_infos(num_workers=workers, has_label=True)
    mmcv.dump(infos_train, train_filename, 'pkl')
    print(f'{pkl_prefix} info train file is saved to {train_filename}')

    infos_val = val_dataset.get_infos(num_workers=workers, has_label=True)
    mmcv.dump(infos_val, val_filename, 'pkl')
    print(f'{pkl_prefix} info val file is saved to {val_filename}')
