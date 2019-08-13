

#include	"particle.h" 

//_____________________________________________________________________________________________________________________________
 
std::random_device				s_RandDevice;
std::minstd_rand				s_Engine(s_RandDevice());
std::normal_distribution<float> s_NormalDist(0, RAND_MAX); 
 
//_____________________________________________________________________________________________________________________________

void PointF3::randomize()
{
	x = s_NormalDist( s_Engine);
	y = s_NormalDist( s_Engine);
	z = s_NormalDist( s_Engine);
}

//_____________________________________________________________________________________________________________________________

TH_UBIQ void PointF3::Normalize()
{
	float t = sqrt(x * x + y * y + z * z);
	x /= t;
	y /= t;
	z /= t;
}

//_____________________________________________________________________________________________________________________________

TH_UBIQ void	PointF3::Scramble()
{
	float	tx = 0.317f * (x + 1.0f) + y + z * x * x + y + z;
	float	ty = 0.619f * (y + 1.0f) + y * y + x * y * z + y + x;
	float	tz = 0.124f * (z + 1.0f) + z * y + x * y * z + y + x;
	x = tx;
	y = ty;
	z = tz;
}
 
//_____________________________________________________________________________________________________________________________

TH_UBIQ void	Particle::Advance(float d)
{
	velocity.Normalize();
	auto	dx = d * velocity.x;
	position.x += dx;
	totalDistance.x += dx;
	auto	dy = d * velocity.y;
	position.y += dy;
	totalDistance.y += dy;
	auto	dz = d * velocity.z;
	position.z += dz;
	totalDistance.z += dz;
	velocity.Scramble();
}

//_____________________________________________________________________________________________________________________________
