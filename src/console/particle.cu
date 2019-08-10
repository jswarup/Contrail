

#include	"particle.h" 

//_____________________________________________________________________________________________________________________________

namespace {
	std::random_device randDevice;
	std::minstd_rand engine(randDevice());
	std::normal_distribution<float> normalDist(0, RAND_MAX);
}

//_____________________________________________________________________________________________________________________________

void randomize(float& x, float& y, float& z)
{
	x = normalDist(engine);
	y = normalDist(engine);
	z = normalDist(engine);
}

//_____________________________________________________________________________________________________________________________

PointF3::PointF3()
{
	::randomize(x, y, z);
}

//_____________________________________________________________________________________________________________________________

PointF3::PointF3(float xIn, float yIn, float zIn) : x(xIn), y(yIn), z(zIn)
{}

//_____________________________________________________________________________________________________________________________

void PointF3::randomize()
{
	::randomize(x, y, z);
}

//_____________________________________________________________________________________________________________________________

__host__ __device__ void PointF3::normalize()
{
	float t = sqrt(x * x + y * y + z * z);
	x /= t;
	y /= t;
	z /= t;
}

//_____________________________________________________________________________________________________________________________

__host__ __device__ void PointF3::scramble()
{
	float	tx = 0.317f * (x + 1.0f) + y + z * x * x + y + z;
	float	ty = 0.619f * (y + 1.0f) + y * y + x * y * z + y + x;
	float	tz = 0.124f * (z + 1.0f) + z * y + x * y * z + y + x;
	x = tx;
	y = ty;
	z = tz;
}

//_____________________________________________________________________________________________________________________________

particle::particle() : 	position(), velocity(), totalDistance(1,0,0)
{
}

//_____________________________________________________________________________________________________________________________

__device__ __host__ void particle::advance(float d)
{
	velocity.normalize();
	auto	dx = d * velocity.x;
	position.x += dx;
	totalDistance.x += dx;
	auto	dy = d * velocity.y;
	position.y += dy;
	totalDistance.y += dy;
	auto	dz = d * velocity.z;
	position.z += dz;
	totalDistance.z += dz;
	velocity.scramble();
}



//_____________________________________________________________________________________________________________________________
