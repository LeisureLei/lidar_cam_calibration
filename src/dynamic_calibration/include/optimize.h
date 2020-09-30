#include <ceres/ceres.h>

struct ReprojectedError
{
    private:
        cv::Point3d p3;
        cv::point2d p2;

    public:
        ReprojectedError(cv::Point3d P3, cv::Point2d P2):p3(P3),p2(P2){}

    template<typename T>
    bool operator()(const T* const q4x1, const T* const t3x1, T* residuals) const{
        Eigen::Quaterniond<T> Rx(q4x1[0], q4x1[1],q4x1[2],q4x1[3]);
        Eigen::Matrix<T,3,1> tx;
        tx<<t3x1[0],t3x1[1],t3x1[2];
  
        Eigen::Vector3d p3_t(Eigen::Vector3d(p3.x, p3.y, p3.z));
        Eigen::Matrix<T,3,1> p3_T = p3_t.cast<T>();
        Eigen::Vector2d p2_t(Eigen::Vector2d(p2.x, p2.y));
        Eigen::Matrix<T,2,1> p2_T = p2_t.cast<T>();

        Eigen::Matrix<T,3,3> K_T = K.cast<T>();

        Eigen::Matrix<T,3,1> Puv = K_T * (Rx.matrix()*p3_T + tx)
        residuals[0] = Puv[0]/Puv[2] - p2_T[0];
        residuals[1] = Puv[1]/Puv[2] - p2_T[1];

        return true;

    }
};