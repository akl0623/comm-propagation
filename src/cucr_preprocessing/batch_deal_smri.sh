#!/bin/bash

# 自动处理sub-02到sub-36的受试者数据
# 需要修改的变量用$subject_id代替

# 设置变量
MNI_TEMPLATE="MNI152_T1_1mm.nii.gz"  # MNI模板文件
HCP_ANNOT_LH="lh.HCP-MMP1.annot"      # 左侧HCP注释文件
HCP_ANNOT_RH="rh.HCP-MMP1.annot"      # 右侧HCP注释文件

# 检查必要的文件是否存在
if [ ! -f "$MNI_TEMPLATE" ]; then
    echo "错误: 找不到MNI模板文件 $MNI_TEMPLATE"
    exit 1
fi

if [ ! -f "$HCP_ANNOT_LH" ] || [ ! -f "$HCP_ANNOT_RH" ]; then
    echo "错误: 找不到HCP注释文件"
    echo "请确保 $HCP_ANNOT_LH 和 $HCP_ANNOT_RH 存在"
    exit 1
fi

# 处理每个受试者
for subject_id in {02..36}; do
    echo "正在处理受试者 sub-$subject_id..."
    
    # 定义输入输出文件
    INPUT_T1="sub-${subject_id}_T1w.nii"
    REORIENTED_T1="sub-${subject_id}_T1w_reoriented.nii.gz"
    MNI_REGISTERED_T1="sub-${subject_id}_T1w_MNI.nii.gz"
    TRANSFORM_MAT="sub-${subject_id}_struct2mni.mat"
    RECON_SUBJECT="sub-${subject_id}_recon"
    OUTPUT_ANNOT_LH="sub-${subject_id}_lh.HCP-MMP1_on_MNI152_ICBM2009a_nlin.annot"
    OUTPUT_ANNOT_RH="sub-${subject_id}_rh.HCP-MMP1_on_MNI152_ICBM2009a_nlin.annot"
    
    # 步骤1: 检查原始T1图像的坐标轴方向
    echo "步骤1: 检查原始图像方向..."
    mri_info "$INPUT_T1" | grep Orientation
    
    # 步骤2: 调整坐标轴方向
    echo "步骤2: 重新定向图像..."
    fslreorient2std "$INPUT_T1" "$REORIENTED_T1"
    
    # 检查重新定向后的方向
    echo "重新定向后的方向:"
    mri_info "$REORIENTED_T1" | grep Orientation
    
    # 步骤3: 刚性配准到MNI空间
    echo "步骤3: 配准到MNI空间..."
    flirt -in "$REORIENTED_T1" -ref "$MNI_TEMPLATE" \
          -out "$MNI_REGISTERED_T1" -omat "$TRANSFORM_MAT" \
          -searchrx -30 30 -searchry -30 30 -searchrz -30 30
    
    # 检查配准后的方向
    echo "配准后的方向:"
    mri_info "$MNI_REGISTERED_T1" | grep Orientation
    
    # 步骤4: 调整x轴方向为RAS
    echo "步骤4: 调整坐标轴方向为RAS..."
    fslorient -swaporient "$MNI_REGISTERED_T1"
    
    # 检查最终方向
    echo "最终方向:"
    mri_info "$MNI_REGISTERED_T1" | grep Orientation
    
    # 步骤5: 运行recon-all进行皮质重建
    echo "步骤5: 运行recon-all (这可能需要很长时间)..."
    recon-all -i "$MNI_REGISTERED_T1" -s "$RECON_SUBJECT" -all
    
    # 步骤6: 映射HCP-MMP1注释到个体表面
    echo "步骤6: 映射HCP-MMP1注释..."
    mri_surf2surf --hemi lh --srcsubject fsaverage --trgsubject "$RECON_SUBJECT" \
                  --sval-annot "$HCP_ANNOT_LH" --tval "$OUTPUT_ANNOT_LH"
    
    mri_surf2surf --hemi rh --srcsubject fsaverage --trgsubject "$RECON_SUBJECT" \
                  --sval-annot "$HCP_ANNOT_RH" --tval "$OUTPUT_ANNOT_RH"
    
    echo "受试者 sub-$subject_id 处理完成!"
    echo "----------------------------------------"
done

echo "所有受试者处理完成!"
