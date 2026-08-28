import fiftyone as fo
import os
import yaml


def convert_yolo_skeleton_to_fiftyone(skeleton):
    """Convert YOLO skeleton format to FiftyOne skeleton format.
    
    YOLO format: list of edges [[1,2], [3,4], [4,5], [5,6], [6,3]]
    FiftyOne format: list of chains [[0,1], [2,3,4,5,2]]
    
    Also converts from 1-indexed (YOLO) to 0-indexed (FiftyOne).
    """
    if not skeleton:
        return []
    
    # skeleton_0indexed = [[edge[0] - 1, edge[1] - 1] for edge in skeleton]
    
    chains = []
    used = set()
    
    for edge in skeleton:
        if tuple(edge) in used:
            continue
        
        chain = [edge[0]]
        current = edge[1]
        chain.append(current)
        used.add(tuple(edge))
        
        while True:
            found_next = False
            for next_edge in skeleton:
                if tuple(next_edge) in used:
                    continue
                if next_edge[0] == current:
                    chain.append(next_edge[1])
                    current = next_edge[1]
                    used.add(tuple(next_edge))
                    found_next = True
                    break
                elif next_edge[1] == current:
                    chain.insert(-1, next_edge[0])
                    current = next_edge[0]
                    used.add(tuple(next_edge))
                    found_next = True
                    break
            if not found_next:
                break
        
        if len(chain) > 1:
            chains.append(chain)
    
    return chains


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    kpt_shape = data.get('kpt_shape', [7, 3])
    num_keypoints = kpt_shape[0]
    kpt_names = data.get('kpt_names', {})
    if 0 in kpt_names and len(kpt_names[0]) >= num_keypoints:
        kp_labels = kpt_names[0][:num_keypoints]
    else:
        kp_labels = [f'kp{i}' for i in range(num_keypoints)]
    
    yolo_skeleton = data.get('skeleton', [])
    fiftyone_skeleton = convert_yolo_skeleton_to_fiftyone(yolo_skeleton)
    
    return {
        'nc': data.get('nc', 15),
        'names': data.get('names', {}),
        'kpt_shape': kpt_shape,
        'skeleton': fiftyone_skeleton,
        'path': data.get('path', '.'),
        'kpt_labels': kp_labels
    }


def load_yolo_pose_annotations(txt_path, num_keypoints):
    annotations = []
    if not os.path.exists(txt_path):
        return annotations

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 + num_keypoints * 3:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            keypoints = []
            for i in range(num_keypoints):
                kp_idx = 5 + i * 3
                kp_x = float(parts[kp_idx])
                kp_y = float(parts[kp_idx + 1])
                kp_v = float(parts[kp_idx + 2])
                keypoints.append([kp_x, kp_y, kp_v])

            annotations.append({
                'class_id': class_id,
                'bbox': [x_center - width/2, y_center - height/2, width, height],
                'keypoints': keypoints
            })
    return annotations


def visualize_baking_dataset():
    yaml_path = 'data/pose-dataset/BakingRecognize/dataset.yaml'
    config = load_config(yaml_path)

    dataset_path = config['path']
    val_file = os.path.join(dataset_path, 'val.txt')

    with open(val_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]

    num_classes = config['nc']
    class_names = config['names']
    num_keypoints = config.get('kpt_shape', [7, 3])[0]
    skeleton = config['skeleton']
    # skeleton = [[p[0]-1, p[1]-1] for p in skeleton0]
    kp_labels = config.get('kpt_labels', [f'kp{i}' for i in range(num_keypoints)])

    dataset_name = 'BakingRecognize_Pose_Validation'
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    for img_path in image_paths:
        if not img_path:
            continue

        img_path = img_path.replace('\\', '/').replace(f'{dataset_path}/', '')
        full_img_path = os.path.join(dataset_path, img_path)

        if not os.path.exists(full_img_path):
            jpg_path = full_img_path.replace('.png', '.jpg')
            if os.path.exists(jpg_path):
                full_img_path = jpg_path
            else:
                print(f'Image not found: {img_path}')
                continue

        txt_path = full_img_path.replace('/images/', '/labels/').replace('.png', '.txt').replace('.jpg', '.txt')

        sample = fo.Sample(filepath=full_img_path)

        if os.path.exists(txt_path):
            annotations = load_yolo_pose_annotations(txt_path, num_keypoints)

            for ann in annotations:
                class_id = ann['class_id']
                class_name = class_names.get(class_id, f'class_{class_id}')
                keypoints = ann['keypoints']

                nonzero_count = sum(1 for kp in keypoints if kp[0] != 0 or kp[1] != 0)

                if nonzero_count <= 3:
                    detection = fo.Detection(
                        label=class_name,
                        bounding_box=ann['bbox']
                    )
                    if not hasattr(sample, 'ground_truth') or sample['ground_truth'] is None:
                        sample['ground_truth'] = fo.Detections(detections=[])
                    sample['ground_truth'].detections.append(detection)

                kp_points = []
                for kp in keypoints:
                    visibility = sum(kp[:2]) != 0 or kp[2]
                    if visibility == 0 or (kp[0] == 0 and kp[1] == 0):
                        kp_points.append([float('nan'), float('nan')])
                    else:
                        kp_points.append([kp[0], kp[1]])

                keypoint = fo.Keypoint(
                    label=class_name,
                    points=kp_points,
                    # confidence=[kp[2] if len(kp) > 2 else 1.0 for kp in keypoints]
                )

                if not hasattr(sample, 'keypoints') or sample['keypoints'] is None:
                    sample['keypoints'] = fo.Keypoints(keypoints=[])
                sample['keypoints'].keypoints.append(keypoint)

        dataset.add_sample(sample)

    if skeleton and len(skeleton) > 0:
        dataset.default_skeleton = fo.KeypointSkeleton(
            labels=kp_labels,
            edges=skeleton
        )

    print(f'Dataset loaded: {len(dataset)} samples')
    print(f'Classes: {num_classes}')
    print(f'Keypoints: {num_keypoints}')
    print(f'Skeleton: {skeleton}')
    print(f'Labels: {kp_labels}')
    print('Launching FiftyOne App...')

    app_config = fo.AppConfig()
    app_config.show_skeletons = True
    session = fo.launch_app(dataset, config=app_config)
    session.wait()


if __name__ == '__main__':
    visualize_baking_dataset()