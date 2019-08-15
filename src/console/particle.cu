

#include	"particle.h" 

//_____________________________________________________________________________________________________________________________
 
std::random_device				s_RandDevice;
std::minstd_rand				s_Engine(s_RandDevice());
std::normal_distribution<float> s_NormalDist(0, RAND_MAX); 
 
//_____________________________________________________________________________________________________________________________

void PointF3::randomize()
{
	X() = s_NormalDist( s_Engine);
	Y() = s_NormalDist( s_Engine);
	Z() = s_NormalDist( s_Engine);
}

//_____________________________________________________________________________________________________________________________

TH_UBIQ void PointF3::Normalize()
{
	float t = sqrt( X() * X() + Y() * Y() + Z() * Z());
	X() /= t;
	Y() /= t;
	Z() /= t;
}

//_____________________________________________________________________________________________________________________________

TH_UBIQ void	PointF3::Scramble()
{
	float	tx = 0.317f * (X() + 1.0f) + Y() + Z() * X() * X() + Y() + Z();
	float	ty = 0.619f * (Y() + 1.0f) + Y() * Y() + X() * Y() * Z() + Y() + X();
	float	tz = 0.124f * (Z() + 1.0f) + Z() * Y() + X() * Y() * Z() + Y() + X();
	X() = tx;
	Y() = ty;
	Z() = tz;
}
 
//_____________________________________________________________________________________________________________________________

TH_UBIQ void	Particle::Advance(float d)
{
	velocity.Normalize();
	auto	dx = d * velocity.X();
	position.X() += dx;
	totalDistance.X() += dx;
	auto	dy = d * velocity.Y();
	position.Y() += dy;
	totalDistance.Y() += dy;
	auto	dz = d * velocity.Z();
	position.Z() += dz;
	totalDistance.Z() += dz;
	velocity.Scramble();
}

//_____________________________________________________________________________________________________________________________
