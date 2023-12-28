use opencv::calib3d::{find_homography, RANSAC};
use opencv::core::{no_array, perspective_transform, Point2f};
use opencv::features2d::{
    draw_keypoints_def, draw_matches_def, draw_matches_knn_def, BFMatcher, FlannBasedMatcher, ORB,
    SIFT,
};
use opencv::flann::{IndexParams, SearchParams, FLANN_INDEX_KDTREE};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::imread_def;
use opencv::imgproc::{cvt_color_def, COLOR_BGR2GRAY};
use opencv::prelude::*;
use opencv::types::{
    PtrOfIndexParams, PtrOfSearchParams, VectorOfDMatch, VectorOfKeyPoint, VectorOfPoint2f,
    VectorOfVectorOfDMatch,
};
use opencv::xfeatures2d::SURF;
use std::process::exit;

/// SIFT特征检测, 无论图片是放大还是缩小, 依然可以将顶角判断出来
///
/// 优点: 检测准确
///
/// 缺点: 效率慢
///
/// 描述子, 记录了关键点周围对其有贡献的像素点的一组向量值, 其不受仿射变换, 光照变换 等影响
fn sift_function() -> opencv::Result<()> {
    let raw_mat = imread_def("C:\\Users\\Administrator\\Desktop\\chess.png")?;

    // sift需要灰度化
    let mut gray = Mat::default();
    cvt_color_def(&raw_mat, &mut gray, COLOR_BGR2GRAY)?;

    // 创建sift对象
    let mut key_point = VectorOfKeyPoint::new();
    let mut sift = SIFT::create_def()?;

    // 检测关键点(位置, 大小和方向)
    // sift.detect_def(&gray, &mut key_point)?;

    // 检测关键点和描述子
    let mut descriptors = Mat::default();
    sift.detect_and_compute_def(&gray, &Mat::default(), &mut key_point, &mut descriptors)?;

    // 绘制关键点
    let mut kp_mat = Mat::default();
    draw_keypoints_def(&raw_mat, &key_point, &mut kp_mat)?;

    imshow("ssd", &kp_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// SURF特征检测, SIFT的改进版本
///
/// 优点: 检测速度快
///
/// 缺点: 准确性没有SIFT高
fn surf_function() -> opencv::Result<()> {
    let raw_mat = imread_def("C:\\Users\\Administrator\\Desktop\\chess.png")?;

    // sift需要灰度化
    let mut gray = Mat::default();
    cvt_color_def(&raw_mat, &mut gray, COLOR_BGR2GRAY)?;

    // 创建sift对象
    let mut key_point = VectorOfKeyPoint::new();
    let mut surf = SURF::create_def()?;

    // 检测关键点(位置, 大小和方向)
    // surf.detect_def(&gray, &mut key_point)?;

    // 检测关键点和描述子
    let mut descriptors = Mat::default();
    surf.detect_and_compute_def(&gray, &Mat::default(), &mut key_point, &mut descriptors)?;

    // 绘制关键点
    let mut kp_mat = Mat::default();
    draw_keypoints_def(&raw_mat, &key_point, &mut kp_mat)?;

    imshow("ssd", &kp_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// ORB特征检测, 无论图片是放大还是缩小, 依然可以将顶角判断出来
///
/// 优点: 可以做到实时检测
///
/// 缺点: 对描述子的数据量进行缩减, 准确性不如SURF和SIFT
fn orb_function() -> opencv::Result<()> {
    let raw_mat = imread_def("C:\\Users\\Administrator\\Desktop\\chess.png")?;

    // sift需要灰度化
    let mut gray = Mat::default();
    cvt_color_def(&raw_mat, &mut gray, COLOR_BGR2GRAY)?;

    // 创建sift对象
    let mut key_point = VectorOfKeyPoint::new();
    let mut orb = ORB::create_def()?;

    // 检测关键点(位置, 大小和方向)
    // surf.detect_def(&gray, &mut key_point)?;

    // 检测关键点和描述子
    let mut descriptors = Mat::default();
    orb.detect_and_compute_def(&gray, &Mat::default(), &mut key_point, &mut descriptors)?;

    // 绘制关键点
    let mut kp_mat = Mat::default();
    draw_keypoints_def(&raw_mat, &key_point, &mut kp_mat)?;

    imshow("ssd", &kp_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// Brute-Force 暴力特征匹配方法
///
/// 优点: 精准匹配
///
/// 原理是, 将A的关键件和描述和B的关键点和描述子进行遍历匹配
///
/// 计算它们之间的差距, 然后将最接近的一个匹配返回
fn bf_function() -> opencv::Result<()> {
    let src_mat = imread_def("C:\\Users\\Administrator\\Desktop\\111.png")?;
    let dst_mat = imread_def("C:\\Users\\Administrator\\Desktop\\222.png")?;

    // sift需要灰度化
    let mut src_gray = Mat::default();
    cvt_color_def(&src_mat, &mut src_gray, COLOR_BGR2GRAY)?;
    let mut dst_gray = Mat::default();
    cvt_color_def(&dst_mat, &mut dst_gray, COLOR_BGR2GRAY)?;

    // 创建sift对象
    let mut sift = SIFT::create_def()?;

    // 关键点
    let mut key_point_src = VectorOfKeyPoint::new();
    let mut key_point_dst = VectorOfKeyPoint::new();
    // 描述子
    let mut descriptors_src = Mat::default();
    let mut descriptors_dst = Mat::default();

    sift.detect_and_compute_def(
        &src_gray,
        &Mat::default(),
        &mut key_point_src,
        &mut descriptors_src,
    )?;
    sift.detect_and_compute_def(
        &dst_gray,
        &Mat::default(),
        &mut key_point_dst,
        &mut descriptors_dst,
    )?;

    // 创建匹配器
    let bf = BFMatcher::create_def()?;
    let mut best_match = VectorOfDMatch::new();
    bf.train_match_def(&descriptors_src, &descriptors_dst, &mut best_match)?;

    // 绘制关键点
    let mut net_mat = Mat::default();
    draw_matches_def(
        &src_mat,
        &key_point_src,
        &dst_mat,
        &key_point_dst,
        &best_match,
        &mut net_mat,
    )?;

    imshow("ssd", &net_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// FLANN 最快近邻区特征匹配方法
///
/// 优点: 效率快, 批量最适合
///
/// 缺点: 匹配不精准
fn flann_function() -> opencv::Result<()> {
    let src_mat = imread_def("C:\\Users\\Administrator\\Desktop\\opencv_search.png")?;
    let dst_mat = imread_def("C:\\Users\\Administrator\\Desktop\\opencv_orig.png")?;

    // sift需要灰度化
    let mut src_gray = Mat::default();
    cvt_color_def(&src_mat, &mut src_gray, COLOR_BGR2GRAY)?;
    let mut dst_gray = Mat::default();
    cvt_color_def(&dst_mat, &mut dst_gray, COLOR_BGR2GRAY)?;

    // 创建sift对象
    let mut sift = SIFT::create_def()?;

    // 关键点
    let mut key_point_src = VectorOfKeyPoint::new();
    let mut key_point_dst = VectorOfKeyPoint::new();
    // 描述子
    let mut descriptors_src = Mat::default();
    let mut descriptors_dst = Mat::default();

    sift.detect_and_compute_def(
        &src_gray,
        &Mat::default(),
        &mut key_point_src,
        &mut descriptors_src,
    )?;
    sift.detect_and_compute_def(
        &dst_gray,
        &Mat::default(),
        &mut key_point_dst,
        &mut descriptors_dst,
    )?;

    // 创建匹配器
    let mut index_params = IndexParams::default()?;
    index_params.set_algorithm(FLANN_INDEX_KDTREE)?;
    index_params.set_int("trees", 5)?;
    let index_params = PtrOfIndexParams::new(index_params);

    let search_params = SearchParams::new_1(50, 0.0, true)?;
    let search_params = PtrOfSearchParams::new(search_params);

    let flann = FlannBasedMatcher::new(&index_params, &search_params)?;

    let mut best_match = VectorOfVectorOfDMatch::new();
    let k = 2; // 查找最优的两个点

    flann.knn_train_match_def(&descriptors_src, &descriptors_dst, &mut best_match, k)?;

    // 过滤比较好的关键点
    let mut result = VectorOfVectorOfDMatch::new();

    for line in &best_match {
        let mut list = VectorOfDMatch::new();

        for singe in line {
            // 值越低, 近似度越高
            if singe.distance < 0.9 {
                list.push(singe);
            }
        }

        result.push(list);
    }

    // 绘制关键点
    let mut net_mat = Mat::default();
    draw_matches_knn_def(
        &src_mat,
        &key_point_src,
        &dst_mat,
        &key_point_dst,
        &result,
        &mut net_mat,
    )?;

    imshow("ssd", &net_mat)?;

    wait_key(100000)?;

    Ok(())
}

/// 单应型矩阵
///
/// 一个图片在不同视角有不同维度, 经过某一点可计算出另一点的位置
fn dan_ying_xing_nv_zhen() -> opencv::Result<()> {
    let src_mat = imread_def("C:\\Users\\Administrator\\Desktop\\opencv_search.png")?;
    let mut dst_mat = imread_def("C:\\Users\\Administrator\\Desktop\\opencv_orig.png")?;

    // sift需要灰度化
    let mut src_gray = Mat::default();
    cvt_color_def(&src_mat, &mut src_gray, COLOR_BGR2GRAY)?;
    let mut dst_gray = Mat::default();
    cvt_color_def(&dst_mat, &mut dst_gray, COLOR_BGR2GRAY)?;

    // 创建sift对象
    let mut sift = SIFT::create_def()?;

    // 关键点
    let mut key_point_src = VectorOfKeyPoint::new();
    let mut key_point_dst = VectorOfKeyPoint::new();
    // 描述子
    let mut descriptors_src = Mat::default();
    let mut descriptors_dst = Mat::default();

    sift.detect_and_compute_def(
        &src_gray,
        &Mat::default(),
        &mut key_point_src,
        &mut descriptors_src,
    )?;
    sift.detect_and_compute_def(
        &dst_gray,
        &Mat::default(),
        &mut key_point_dst,
        &mut descriptors_dst,
    )?;

    // 创建匹配器
    let mut index_params = IndexParams::default()?;
    index_params.set_algorithm(FLANN_INDEX_KDTREE)?;
    index_params.set_int("trees", 5)?;
    let index_params = PtrOfIndexParams::new(index_params);

    let search_params = SearchParams::new_1(50, 0.0, true)?;
    let search_params = PtrOfSearchParams::new(search_params);

    let flann = FlannBasedMatcher::new(&index_params, &search_params)?;

    let mut best_match = VectorOfVectorOfDMatch::new();
    let k = 2; // 查找最优的两个点

    flann.knn_train_match_def(&descriptors_src, &descriptors_dst, &mut best_match, k)?;

    // 过滤比较好的关键点
    let mut result = VectorOfVectorOfDMatch::new();

    for line in &best_match {
        let mut list = VectorOfDMatch::new();

        for singe in line {
            // 值越低, 近似度越高
            if singe.distance < 0.9 {
                list.push(singe);
            }
        }

        result.push(list);
    }

    if best_match.len() >= 4 {
        // 单应性矩阵, 意思就是将图A在图B中用矩形圈起来
        let mut src_pts = VectorOfPoint2f::new();
        let mut dst_pts = VectorOfPoint2f::new();
        for key_point in best_match {
            for elem in key_point {
                let query_idx = key_point_src.get(elem.query_idx as usize)?;
                src_pts.push(query_idx.pt());

                let train_idx = key_point_dst.get(elem.train_idx as usize)?;
                dst_pts.push(train_idx.pt());
            }
        }

        // 随机抽样, 经验值5
        let mut h = find_homography(&src_pts, &dst_pts, &mut no_array(), RANSAC, 5f64)?;

        let weight = h.size()?.width;
        let height = h.size()?.height;

        let mut pts = VectorOfPoint2f::new();
        pts.push(Point2f::new(0f32, 0f32));
        pts.push(Point2f::new(0f32, (height - 1) as f32));
        pts.push(Point2f::new((weight - 1) as f32, (height - 1) as f32));
        pts.push(Point2f::new((weight - 1) as f32, 0f32));

        perspective_transform(&pts, &mut h, &no_array())?;

        // polylines_def(&mut dst_mat, &pts, true, Scalar::from((0, 0, 255)))?;
        //
        // // 绘制关键点
        // let mut net_mat = Mat::default();
        // draw_matches_knn_def(
        //     &src_mat,
        //     &key_point_src,
        //     &dst_mat,
        //     &key_point_dst,
        //     &result,
        //     &mut net_mat,
        // )?;
        //
        imshow("ssd", &h)?;

        wait_key(100000)?;
    } else {
        println!("数组长度不能小于4个");
        exit(0)
    }

    Ok(())
}

fn main() -> opencv::Result<()> {
    // sift_function()?;
    // surf_function()?;
    // orb_function()?;
    // bf_function()?;
    // flann_function()?;
    dan_ying_xing_nv_zhen()?;

    Ok(())
}
