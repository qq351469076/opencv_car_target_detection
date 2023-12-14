use opencv::bgsegm::create_background_subtractor_mog_def;
use opencv::core::{Point, Rect, Scalar, Size, BORDER_CONSTANT, BORDER_DEFAULT};
use opencv::highgui::{destroy_all_windows, imshow, wait_key};
use opencv::imgproc::{
    bounding_rect, cvt_color_def, dilate, erode_def, find_contours_def, gaussian_blur_def,
    get_structuring_element_def, line_def, morphology_ex_def, rectangle_def, CHAIN_APPROX_SIMPLE,
    COLOR_BGR2GRAY, MORPH_CLOSE, MORPH_RECT, RETR_TREE,
};
use opencv::prelude::*;
use opencv::types::VectorOfVectorOfPoint;
use opencv::videoio::VideoCapture;

const MIN_W: i32 = 90;
const MIN_H: i32 = 90;

const LINE_HIGH: i32 = 600;

const LINE_OFFSET: i32 = 6;

fn center(point: &Rect) -> Point {
    let x = point.width / 2;
    let y = point.height / 2;

    let center_x = point.x + x;
    let center_y = point.y + y;

    Point {
        x: center_x,
        y: center_y,
    }
}

fn main() -> opencv::Result<()> {
    let mut capture = VideoCapture::from_file_def("C:\\Users\\Administrator\\Desktop\\video.mp4")?;

    // 去后背景, 参数history:200, 如果视频是25帧, 大概取8帧图片作为历史, 只有像素点发生移动才会认为是前背景
    let mut bgsubmog = create_background_subtractor_mog_def()?;

    let kernel = get_structuring_element_def(MORPH_RECT, Size::new(5, 5))?;

    let mut cars = Vec::new();

    let mut car_num = 0;

    loop {
        // 读取帧
        let mut frame = Mat::default();
        let ret = capture.read(&mut frame)?;

        if !ret {
            break;
        }

        // 转灰度
        let mut cvt_frame = Mat::default();
        cvt_color_def(&frame, &mut cvt_frame, COLOR_BGR2GRAY)?;

        // 去噪
        let mut blur = Mat::default();
        gaussian_blur_def(&cvt_frame, &mut blur, Size::new(3, 3), 5f64)?;

        // 去背景
        let mut mask = Mat::default();
        bgsubmog.apply_def(&blur, &mut mask)?;

        // 腐蚀(在去掉一些小噪点)
        let mut erode_mat = Mat::default();
        erode_def(&mask, &mut erode_mat, &kernel)?;

        // 经过腐蚀, 噪点没有, 图片变小了, 在通过膨胀放大到原来
        let mut dilate_mat = Mat::default();
        dilate(
            &erode_mat,
            &mut dilate_mat,
            &kernel,
            Point::new(-1, -1),
            3,
            BORDER_CONSTANT,
            Scalar::from(BORDER_DEFAULT),
        )?;

        // 闭运算,去掉物内部小块
        let mut close_mat = Mat::default();
        morphology_ex_def(&dilate_mat, &mut close_mat, MORPH_CLOSE, &kernel)?;
        let mut close_mat_1 = Mat::default();
        morphology_ex_def(&close_mat, &mut close_mat_1, MORPH_CLOSE, &kernel)?;

        // 查找轮廓
        let mut contours = VectorOfVectorOfPoint::new();
        find_contours_def(&close_mat_1, &mut contours, RETR_TREE, CHAIN_APPROX_SIMPLE)?;

        // 画一条检测线
        line_def(
            &mut frame,
            Point::new(10, LINE_HIGH),
            Point::new(1200, LINE_HIGH),
            (0, 0, 255).into(),
        )?;

        for (index, list) in contours.iter().enumerate() {
            let rect = bounding_rect(&list)?;

            // 过滤不是车的物体
            if rect.width < MIN_W && rect.height < MIN_H {
                continue;
            }

            // 获取车辆的中心点
            let car_center_point = center(&rect);
            cars.push(car_center_point);

            for car in &mut cars {
                // 要有一条线,并且有范围, 上下6个像素
                if car.y > (LINE_HIGH - LINE_OFFSET) && (car.y < (LINE_HIGH + LINE_OFFSET)) {
                    car_num += 1;
                    // cars.remove(1);
                    println!("{:?}", car_num)
                }
            }

            rectangle_def(&mut frame, rect, Scalar::from((0, 0, 255)))?;
        }
        imshow("adas", &frame)?;

        let key = wait_key(25)?;
        if key > 0 && key != 255 {
            break;
        }
    }

    capture.release()?;

    destroy_all_windows()?;

    Ok(())
}
