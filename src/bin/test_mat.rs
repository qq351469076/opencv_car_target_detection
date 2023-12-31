use opencv::core::{
    self, MatConstIterator, MatIter, Point, Point2d, Rect, Scalar, Size, Vec3d, Vec3f, Vec4w,
};
use opencv::prelude::*;
use opencv::types::VectorOfi32;
use std::ffi::c_void;

const PIXEL: &[u8] = include_bytes!("pixel.png");

/// 矩阵的基本属性
fn mat_default() -> opencv::Result<()> {
    let mat = Mat::default();

    println!("矩阵元素类型为{:?}", mat.typ());
    println!("矩阵元素类型大小为{:?}", mat.depth());
    println!("矩阵元素通道数为{:?}", mat.channels());
    println!("矩阵元素大小为{:?}", mat.size()?);
    println!("矩阵元素维度为{:?}", mat.dims());

    assert_eq!(u8::opencv_type(), mat.typ());
    assert_eq!(u8::opencv_depth(), mat.depth());
    assert_eq!(u8::opencv_channels(), mat.channels());
    assert_eq!(Size::new(0, 0), mat.size()?);
    assert_eq!(0, mat.dims());
    assert!(!mat.is_allocated());
    assert!(mat.data().is_null());
    Ok(())
}

/// 创建x行x列的二维数组(unsafe), 并获取指定位置
/// 用于在给定的Mat对象上动态分配新的数组数据。它使用了一种机制来管理内存，通过释放以前的数据并为新的数据分配内存。它返回一个结果来指示分配是否成功。
fn mat_create() -> opencv::Result<()> {
    let mut mat = Mat::default();
    // 创建10行10列的u16矩阵
    unsafe { mat.create_rows_cols(10, 10, u16::opencv_type())? };
    assert!(mat.is_allocated());
    assert!(!mat.data().is_null());
    // 将所有元素设置成7
    mat.set_scalar(7.into())?;
    // 获取指定2维数组指定位置的元素
    assert_eq!(7, *mat.at_2d::<u16>(0, 0)?);
    assert_eq!(7, *mat.at_2d::<u16>(3, 3)?);
    assert_eq!(7, *mat.at_2d::<u16>(9, 9)?);
    mat.release()?;
    assert!(!mat.is_allocated());
    assert!(mat.data().is_null());
    assert_eq!(Size::new(0, 0), mat.size()?);
    Ok(())
}

/// 从VectorOfi32或vec数组中创建二维Mat, 并获取指定指定位置元素
fn mat_from_iter() -> opencv::Result<()> {
    {
        let mut vec = VectorOfi32::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        let mat = Mat::from_exact_iter(vec.into_iter())?;
        println!("有{:?}行", mat.rows());
        println!("有{:?}列", mat.cols());
        assert_eq!(3, mat.rows());
        assert_eq!(1, mat.cols());
        assert_eq!(i32::opencv_type(), mat.typ());
        println!("元素类型为{:?}", mat.typ());
        assert_eq!(1, *mat.at_2d::<i32>(0, 0)?);
        assert_eq!(2, *mat.at_2d::<i32>(1, 0)?);
        assert_eq!(3, *mat.at_2d::<i32>(2, 0)?);
    }

    {
        let vec: Vec<i32> = vec![1, 2, 3];
        let mat = Mat::from_exact_iter(vec.into_iter())?;
        assert_eq!(3, mat.rows());
        assert_eq!(1, mat.cols());
        assert_eq!(i32::opencv_type(), mat.typ());
        assert_eq!(1, *mat.at_2d::<i32>(0, 0)?);
        assert_eq!(2, *mat.at_2d::<i32>(1, 0)?);
        assert_eq!(3, *mat.at_2d::<i32>(2, 0)?);
    }
    Ok(())
}

/// Mat类的构造函数之一。
/// 这个函数使用了一种简化的方式来创建一个新的Mat对象，使用指定的行数、列数和数据类型来初始化新的矩阵。
/// 它返回一个Mat对象。
fn mat_for_rows_and_cols() -> opencv::Result<()> {
    let mat = unsafe { Mat::new_rows_cols(400, 300, Vec3d::opencv_type()) }?;
    assert_eq!(Vec3d::opencv_type(), mat.typ());
    assert_eq!(Vec3d::opencv_depth(), mat.depth());
    println!("{:?}", mat.channels());
    assert_eq!(Vec3d::opencv_channels(), mat.channels());
    assert!(mat.is_continuous());
    assert!(!mat.is_submatrix());
    assert_eq!(Size::new(300, 400), mat.size()?);
    assert_eq!(400, mat.rows());
    assert_eq!(300, mat.cols());
    assert_eq!(2, mat.mat_size().len());
    assert_eq!(400, mat.mat_size()[0]);
    assert_eq!(300, mat.mat_size()[1]);
    assert_eq!(2, mat.dims());
    assert_eq!(2, mat.mat_step().buf().len());
    assert_eq!(7200, mat.mat_step().buf()[0]);
    assert_eq!(24, mat.mat_step().buf()[1]);
    assert_eq!(24, mat.elem_size()?);
    assert_eq!(8, mat.elem_size1());
    assert_eq!(900, mat.step1(0)?);
    assert_eq!(3, mat.step1(1)?);
    assert_eq!(120000, mat.total());
    Ok(())
}

/// 创建指定维度x行x列的数组
fn mat_nd() -> opencv::Result<()> {
    {
        // 创建指定维度x行x列的数组, 3行, 3列 类型为Vec4w类型, 全部由0填充的矩阵
        let mut mat = Mat::new_nd_with_default(&[3, 3, 3], Vec4w::opencv_type(), 0.into())?;

        // 打印 第一维度的第1行第1列的元素
        println!("{:?}", mat.at_3d::<Vec4w>(0, 1, 1)?);

        assert_eq!(0, mat.at_3d::<Vec4w>(1, 1, 1)?[0]);
        // 修改第一维度, 第一行, 第一列的元素, 将其全部设置成5
        *mat.at_3d_mut::<Vec4w>(1, 1, 1)? = Vec4w::all(10);
        assert_eq!(10, mat.at_3d::<Vec4w>(1, 1, 1)?[0]);
        assert_eq!(0, mat.at_3d::<Vec4w>(1, 1, 2)?[2]);
        assert_eq!(3, mat.dims());
        assert_eq!([3, 3, 3], *mat.mat_size());
    }

    {
        // 矩阵的长度是5040是因为您创建的矩阵（mat）是一个多维矩阵，其维度为[2, 3, 4, 5, 6, 7]。该维度的所有元素相乘得到矩阵的长度
        let dims = VectorOfi32::from_iter(vec![2, 3, 4, 5, 6, 7]);
        let mut mat = Mat::new_nd_vec_with_default(&dims, Vec4w::opencv_type(), 0.into())?;

        assert_eq!(-1, mat.rows());
        assert_eq!(-1, mat.cols());

        assert_eq!(Vec4w::default(), *mat.at_nd::<Vec4w>(&[1, 2, 3, 4, 5, 6])?);
        // 索引表示矩阵中某个元素的位置，其中第1行，第2列，第3深度，第4高度，第5宽度和第6通道。
        *mat.at_nd_mut::<Vec4w>(&[1, 2, 3, 4, 5, 6])? = Vec4w::from([5, 6, 7, 8]);
        assert_eq!(
            Vec4w::from([5, 6, 7, 8]),
            *mat.at_nd::<Vec4w>(&[1, 2, 3, 4, 5, 6])?
        );
    }

    Ok(())
}

/// 从数组引用里创建二维数组, 并且改变矩阵的形状
fn mat_at_1d() -> opencv::Result<()> {
    let s: Vec<Vec<f32>> = vec![vec![1., 2., 3.], vec![4., 5., 6.], vec![7., 8., 9.]];

    {
        // 从数组切片中创建二维数组,  3行3列
        let mat = Mat::from_slice_2d(&s)?;

        // 1个通道, 拆分成1行, 相当于1行9列
        let mut mat = mat.reshape(1, 1)?;

        assert_eq!(1, mat.rows());
        assert_eq!(9, mat.cols());

        assert_eq!(*mat.at::<f32>(0)?, 1.);
        assert_eq!(*mat.at::<f32>(5)?, 6.);
        assert_eq!(*mat.at::<f32>(8)?, 9.);
        *mat.at_mut::<f32>(4)? = 2.;
        assert_eq!(*mat.at::<f32>(4)?, 2.);
    }

    {
        let mat = Mat::from_slice_2d(&s)?;
        // 1个通道, 拆分成9行, 相当于9行1列
        let mut mat = mat.reshape(1, 9)?;
        assert_eq!(9, mat.rows());
        assert_eq!(1, mat.cols());

        assert_eq!(*mat.at::<f32>(0)?, 1.);
        assert_eq!(*mat.at::<f32>(4)?, 5.);
        assert_eq!(*mat.at::<f32>(8)?, 9.);
        *mat.at_mut::<f32>(4)? = 2.;
        assert_eq!(*mat.at::<f32>(4)?, 2.);
    }

    {
        let mut mat = Mat::from_slice_2d(&s)?;

        assert_eq!(*mat.at::<f32>(0)?, 1.);
        assert_eq!(*mat.at::<f32>(6)?, 7.);
        assert_eq!(*mat.at::<f32>(8)?, 9.);
        *mat.at_mut::<f32>(4)? = 2.;
        assert_eq!(*mat.at::<f32>(4)?, 2.);
    }
    Ok(())
}

/// 通过Size结构体创建二维数组
fn mat_2d_i0_is_rows_i1_is_cols() -> opencv::Result<()> {
    // Just a sanity check about which Mat dimension corresponds to which in Size
    let mat = Mat::new_size_with_default(Size::new(6, 5), f32::opencv_type(), 1.23.into())?;
    let size = mat.size()?;
    assert_eq!(size.width, 6);
    assert_eq!(size.height, 5);
    Ok(())
}

/// 指定行和列去创建二维数组, 并且修改某个位置的具体的单个值
fn mat_at_2d() -> opencv::Result<()> {
    let mut mat = Mat::new_rows_cols_with_default(100, 100, f32::opencv_type(), 1.23.into())?;
    assert_eq!(*mat.at_2d::<f32>(0, 0)?, 1.23);
    *mat.at_2d_mut::<f32>(0, 0)? = 1.;
    assert_eq!(*mat.at_2d::<f32>(0, 0)?, 1.);

    Ok(())
}

/// 修改二维数组中某个元素的列表值, 从数组中进行赋值
fn mat_at_2d_multichannel() -> opencv::Result<()> {
    let mut mat =
        Mat::new_rows_cols_with_default(100, 100, Vec3f::opencv_type(), Scalar::all(1.23))?;
    let pix = *mat.at_2d::<Vec3f>(0, 0)?;
    assert_eq!(pix[0], 1.23);
    assert_eq!(pix[1], 1.23);
    assert_eq!(pix[2], 1.23);

    // 修改某个具体位置的值
    *mat.at_2d_mut::<Vec3f>(0, 0)? = Vec3f::from([1.1, 2.2, 3.3]);

    let pix = *mat.at_2d::<Vec3f>(0, 0)?;
    assert_eq!(pix[0], 1.1);
    assert_eq!(pix[1], 2.2);
    assert_eq!(pix[2], 3.3);

    Ok(())
}

/// 创建x行x列的二维数组, 并且修改某一连续的索引的值从数组引用中进行赋值
fn mat_at_row() -> opencv::Result<()> {
    let mut mat = Mat::new_rows_cols_with_default(100, 100, f32::opencv_type(), 1.23.into())?;

    // 选取第一行
    let row = mat.at_row::<f32>(0)?;
    assert_eq!(row.len(), 100);
    assert_eq!(row[0], 1.23);

    // 选取第二行
    // 对第二行的从索引0到索引索引3的位置进行赋值, 从数组引用中进行赋值
    let row = mat.at_row_mut::<f32>(1)?;
    row[0..4].copy_from_slice(&[10., 20., 30., 40.]);

    let data = mat.data_typed::<f32>()?;
    assert_eq!(data[0], 1.23);
    assert_eq!(data[100], 10.);
    assert_eq!(data[101], 20.);
    assert_eq!(data[102], 30.);
    assert_eq!(data[103], 40.);

    Ok(())
}

/// 通过Size结构体来索引位置
fn mat_at_pt() -> opencv::Result<()> {
    // 三行三列
    let s: Vec<Vec<f32>> = vec![vec![1., 2., 3.], vec![4., 5., 6.], vec![7., 8., 9.]];
    let mut m = Mat::from_slice_2d(&s)?;

    assert_eq!(5., *m.at_pt::<f32>(Point::new(1, 1))?);
    assert_eq!(4., *m.at_pt_mut::<f32>(Point::new(0, 1))?);
    assert_eq!(3., unsafe { *m.at_pt_unchecked::<f32>(Point::new(2, 0))? });
    assert_eq!(9., unsafe {
        *m.at_pt_unchecked_mut::<f32>(Point::new(2, 2))?
    });

    Ok(())
}

/// 将Mat转换成数组
///
/// 通过数组构建ndarray, 并且修改某个通道的值
fn mat_vec() -> opencv::Result<()> {
    {
        // 三行三列
        let s: Vec<Vec<f32>> = vec![vec![1., 2., 3.], vec![4., 5., 6.], vec![7., 8., 9.]];

        let mat = Mat::from_slice_2d(&s)?;
        assert_eq!(
            mat.size()?,
            Size {
                width: 3,
                height: 3
            }
        );
        assert_eq!(*mat.at_2d::<f32>(1, 1)?, 5.);

        // 将Mat转换成数组
        let v = mat.to_vec_2d::<f32>()?;

        println!("{:?}", v);

        assert_eq!(s, v);
    }

    {
        {
            let mut dims = VectorOfi32::new();
            // 维度, 此处为深度3, 相当于3个通道
            dims.push(3);
            dims.push(3);
            dims.push(3);
            let mut mat = Mat::new_nd_vec_with_default(&dims, f64::opencv_type(), 2.into())?;
            // 深度0 = 通道1   深度1 = 通道2   深度2 = 通道3
            *mat.at_3d_mut::<f64>(0, 1, 1)? = 10.;

            assert_eq!(3, mat.dims());
            if mat.to_vec_2d::<f64>().is_ok() {
                panic!("dims too high");
            }
        }
    }

    Ok(())
}

fn mat_continuous() {}

fn mat_merge_split() {}

/// 从已加载的图片字节数组中读取图片并转换成Mat
fn mat_from_data() -> opencv::Result<()> {
    let mut bytes = PIXEL.to_vec();
    assert_eq!(90, bytes.len());

    {
        let src = unsafe {
            Mat::new_rows_cols_with_data(
                1,
                PIXEL.len().try_into()?,
                u8::opencv_type(),
                bytes.as_mut_ptr().cast::<c_void>(),
                core::Mat_AUTO_STEP,
            )?
        };
        assert_eq!(Size::new(PIXEL.len().try_into()?, 1), src.size()?);
        assert_eq!(PIXEL.len(), src.total());
        let row = src.at_row::<u8>(0)?;
        assert_eq!(0x89, row[0]);
        assert_eq!(0x50, row[1]);
        assert_eq!(0x1A, row[6]);
        assert_eq!(0x0D, row[11]);
        assert_eq!(0x82, row[89]);
    }

    {
        let src = unsafe {
            // 这个数组唯独是3 * 5 * 6 = 90, 正好对应图片长度
            Mat::new_nd_with_data(
                &[3, 5, 6],
                u8::opencv_type(),
                bytes.as_mut_ptr().cast::<c_void>(),
                None,
            )?
        };
        assert_eq!(Size::new(5, 3), src.size()?);
        assert_eq!(PIXEL.len(), src.total());
        assert_eq!(0x89, *src.at_3d::<u8>(0, 0, 0)?);
        assert_eq!(0x50, *src.at_3d::<u8>(0, 0, 1)?);
        assert_eq!(0x1A, *src.at_3d::<u8>(0, 1, 0)?);
        assert_eq!(0x0D, *src.at_3d::<u8>(0, 1, 5)?);
        assert_eq!(0x82, *src.at_3d::<u8>(2, 4, 5)?);
    }

    {
        let mut bytes = bytes.clone();
        let mut mat = unsafe {
            Mat::new_rows_cols_with_data(
                1,
                bytes.len().try_into()?,
                u8::opencv_type(),
                bytes.as_mut_ptr().cast::<c_void>(),
                core::Mat_AUTO_STEP,
            )?
        };
        assert_eq!(mat.data(), bytes.as_ptr());
        bytes[0] = 0xFF;
        assert_eq!(0xFF, *mat.at::<u8>(0)?);
        mat.resize_with_default(100, 0.into())?;
        assert_ne!(mat.data(), bytes.as_ptr());
        bytes[0] = 0xAA;
        let row = mat.at_row::<u8>(0)?;
        assert_eq!(0xFF, row[0]);
        assert_eq!(0x50, row[1]);
        assert_eq!(0x1A, row[6]);
        assert_eq!(0x0D, row[11]);
        assert_eq!(0x82, row[89]);
        let row = mat.at_row::<u8>(1)?;
        assert_eq!(0, row[1]);
        assert_eq!(0, row[6]);
        assert_eq!(0, row[89]);
    }

    Ok(())
}

/// 创建全0数组
///
/// 创建全1数组
fn mat_from_matexpr() -> opencv::Result<()> {
    {
        let mat = Mat::zeros(3, 4, f32::opencv_type())?.to_mat()?;
        assert_eq!(4, mat.cols());
        assert_eq!(3, mat.rows());
        assert_eq!(0., *mat.at_2d::<f32>(0, 0)?);
        assert_eq!(0., *mat.at_2d::<f32>(1, 1)?);
        assert_eq!(0., *mat.at_2d::<f32>(2, 3)?);
    }

    {
        let mat = Mat::ones_nd(&[6, 5], f32::opencv_type())?.to_mat()?;
        assert_eq!(5, mat.cols());
        assert_eq!(6, mat.rows());
        assert_eq!(1., *mat.at_2d::<f32>(0, 0)?);
        assert_eq!(1., *mat.at_2d::<f32>(1, 1)?);
        assert_eq!(1., *mat.at_2d::<f32>(5, 4)?);
    }

    Ok(())
}

/// 从切片中创建一维Mat, 迭代Mat, 查看当前迭代器坐标, 判断是否还有元素
fn mat_const_iterator() -> opencv::Result<()> {
    {
        let mat = Mat::from_slice(&[1, 2, 3, 4])?;
        let mut iter = MatConstIterator::over(&mat)?;
        assert_eq!(iter.typ(), mat.typ());
        // 推动迭代器
        assert_eq!(1, *iter.current::<i32>()?);
        // 查看当前的坐标
        assert_eq!(Point::new(0, 0), iter.pos()?);
        // 判断之后是否还有元素
        assert!(iter.has_elements());

        // 定位到特定位置的元素
        iter.seek(1, true)?;
        assert_eq!(2, *iter.current::<i32>()?);
        assert_eq!(Point::new(1, 0), iter.pos()?);
        assert!(iter.has_elements());

        iter.seek(1, true)?;
        assert_eq!(3, *iter.current::<i32>()?);
        assert_eq!(Point::new(2, 0), iter.pos()?);
        assert!(iter.has_elements());

        iter.seek(1, true)?;
        assert_eq!(4, *iter.current::<i32>()?);
        assert_eq!(Point::new(3, 0), iter.pos()?);
        assert!(iter.has_elements());

        iter.seek(1, true)?;
        assert_eq!(Point::new(0, 1), iter.pos()?);
        assert!(!iter.has_elements());

        iter.seek(1, true)?;
        assert_eq!(Point::new(0, 1), iter.pos()?);
        assert!(!iter.has_elements());
    }

    Ok(())
}

/// 创建二维Mat, 迭代二维Mat
///
/// 指定从某个坐标开始迭代
///
/// 边迭代, 边修改值
fn mat_iterator() -> opencv::Result<()> {
    {
        let mat = Mat::from_slice_2d(&[[1, 2], [3, 4]])?;
        for (pos, x) in mat.iter::<i32>()? {
            match pos {
                Point { x: 0, y: 0 } => assert_eq!(x, 1),
                Point { x: 1, y: 0 } => assert_eq!(x, 2),
                Point { x: 0, y: 1 } => assert_eq!(x, 3),
                Point { x: 1, y: 1 } => assert_eq!(x, 4),
                _ => panic!("Too many elements"),
            }
        }

        // 指定从哪个坐标开始迭代
        for (pos, x) in MatIter::<i32>::new(MatConstIterator::with_start(&mat, Point::new(1, 0))?)?
        {
            match pos {
                Point { x: 1, y: 0 } => assert_eq!(x, 2),
                Point { x: 0, y: 1 } => assert_eq!(x, 3),
                Point { x: 1, y: 1 } => assert_eq!(x, 4),
                _ => panic!("Too many elements"),
            }
        }
    }

    {
        // 二维矩阵
        let mat = Mat::from_slice_2d(&[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ])?;

        // 从 (1, 1)处选取宽2高2的矩阵
        let roi = Mat::roi(&mat, Rect::new(1, 1, 2, 2))?;
        for (pos, x) in roi.iter::<i32>()? {
            match pos {
                Point { x: 0, y: 0 } => assert_eq!(x, 6),
                Point { x: 1, y: 0 } => assert_eq!(x, 7),
                Point { x: 0, y: 1 } => assert_eq!(x, 10),
                Point { x: 1, y: 1 } => assert_eq!(x, 11),
                _ => panic!("Too many elements"),
            }
        }
    }

    {
        let mut mat = Mat::from_slice_2d(&[[1, 2], [3, 4]])?;
        // 边迭代, 边修改值
        for (pos, x) in mat.iter_mut::<i32>()? {
            *x *= pos.x + pos.y;
        }
        assert_eq!([0, 2, 3, 8], mat.data_typed::<i32>()?);
    }

    {
        let mat = Mat::from_slice_2d(&[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ])?;
        let mut roi = Mat::roi(&mat, Rect::new(1, 1, 2, 2))?;
        for (pos, x) in roi.iter_mut::<i32>()? {
            *x += pos.x + pos.y;
        }
        assert_eq!(
            [1, 2, 3, 4, 5, 6, 8, 8, 9, 11, 13, 12, 13, 14, 15, 16],
            mat.data_typed::<i32>()?
        );
    }

    {
        let mat = Mat::from_slice::<u8>(&[])?;
        #[allow(clippy::never_loop)]
        for _ in mat.iter::<u8>()? {
            panic!("Mat must be empty");
        }
    }

    Ok(())
}

/// 通过ROI提取渔区之后, 存储当前Size和Point
fn mat_locate_roi() -> opencv::Result<()> {
    let mat = Mat::from_slice(&[1, 2, 3, 4])?;
    let roi = Mat::roi(&mat, Rect::new(1, 0, 2, 1))?;
    let mut sz = Size::default();
    println!("{:?}", sz);
    let mut ofs = Point::default();
    println!("{:?}", ofs);
    roi.locate_roi(&mut sz, &mut ofs)?;
    assert_eq!(sz, Size::new(4, 1));
    assert_eq!(ofs, Point::new(1, 0));

    println!("{:?}", sz);
    println!("{:?}", ofs);

    Ok(())
}

/// 转换矩阵的类型
fn mat_convert() -> opencv::Result<()> {
    let mat = Mat::from_slice(&[1, 2, 3, 4])?;
    let mut mat_ = mat.try_clone()?.try_into_typed::<i32>()?;
    assert_eq!(3, *mat_.at(2)?);
    *mat_.at_mut(3)? = 8;
    assert_eq!(8, *mat_.at(3)?);
    assert_eq!(mat.typ(), mat_.typ());
    assert_eq!(mat.size()?, mat_.size()?);
    let mat_back = mat_.into_untyped();
    assert_eq!(mat.size()?, mat_back.size()?);
    Ok(())
}

fn mat_mul() {}

fn mat_data() {}

fn mat_equals() {}

fn mat_rgb() {}

/// 从数组引用中创建Mat, 按照x行x列
fn mat_from_slice() -> opencv::Result<()> {
    let src_u8 = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let src_point = [
        Point2d::new(10.1, 20.2),
        Point2d::new(30.3, 40.4),
        Point2d::new(50.5, 60.6),
        Point2d::new(70.7, 80.8),
    ];

    {
        let mat = Mat::from_slice(&src_u8)?;
        assert_eq!(10, mat.total());
        assert_eq!(5u8, *mat.at(5)?);

        let mat = Mat::from_slice(&src_point)?;
        assert_eq!(4, mat.total());
        assert_eq!(Point2d::new(30.3, 40.4), *mat.at(1)?);
    }

    {
        // 从数组引用中创建Mat, 按照x行x列
        let mat = Mat::from_slice_rows_cols(&src_u8, 2, 5)?;
        assert_eq!(10, mat.total());
        assert_eq!(2, mat.rows());
        assert_eq!(5, mat.cols());
        assert_eq!(6u8, *mat.at_2d(1, 1)?);

        let mat = Mat::from_slice_rows_cols(&src_point, 2, 2)?;
        assert_eq!(4, mat.total());
        assert_eq!(2, mat.rows());
        assert_eq!(2, mat.cols());
        assert_eq!(Point2d::new(50.5, 60.6), *mat.at_2d(1, 0)?);
    }

    Ok(())
}

fn main() -> opencv::Result<()> {
    // // 矩阵的基本属性
    // mat_default()?;

    // // 创建x行x列的二维数组(unsafe), 并获取指定位置
    // mat_create()?;

    // // 从VectorOfi32或vec数组中创建二维Mat, 并获取指定指定位置元素
    // mat_from_iter()?;

    // // C++原生的创建Mat方法, 执行x行x列创建Mat
    // mat_for_rows_and_cols()?;

    // // 创建多维x行x列的数组
    // mat_nd()?;

    // // 从数组引用里创建Mat, 并且改变矩阵的形状
    // mat_at_1d()?;

    // 通过Size结构体创建二维数组
    // mat_2d_i0_is_rows_i1_is_cols()?;

    // // 指定行和列去创建二维数组, 并且修改某个位置的具体的单个值
    // mat_at_2d()?;

    // // 修改二维数组中某个元素的列表值, 从数组中进行赋值
    // mat_at_2d_multichannel()?;

    // // 创建x行x列的二维数组, 并且修改某一连续的索引的值从数组引用中进行赋值
    // mat_at_row()?;

    // 通过Size结构体来索引位置
    // mat_at_pt()?;

    // // 将Mat转换成数组
    // // 通过数组构建ndarray, 并且修改某个通道的值
    // mat_vec()?;

    // // 从已加载的图片字节数组中读取图片并转换成Mat
    // mat_from_data()?;

    // 创建全0数组
    // 创建全1数组
    // mat_from_matexpr()?;

    // // 从切片中创建Mat, 迭代Mat, 查看当前迭代器坐标, 判断是否还有元素
    // mat_const_iterator()?;

    // 创建二维Mat, 迭代二维Mat
    // 指定从某个坐标开始迭代
    // 边迭代, 边修改值
    // mat_iterator()?;

    // // 通过ROI提取渔区之后, 存储当前Size和Point
    // mat_locate_roi()?;

    //
    mat_convert()?;

    // 从数组引用中创建Mat, 按照x行x列
    mat_from_slice()?;

    Ok(())
}
