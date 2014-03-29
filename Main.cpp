#include "Training.h"
#include "Classifiers.h"
#include "FaceDetector.h"
#include "Image.h"
#include "Tracker.h"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"


int main(int argc, char* argv[])
{
	std::string pPath("D:/Training Data/GENKIVJ");
	std::string nPath("D:/Training Data/Negatives");
	std::string valPath("C:/Users/Cihan/Desktop/MIT+CMU_Test");
	uint numP = 8400, numN = 40000, T = 200;
	std::vector<float> tFA;
	std::vector<uint> numAPos;

	float _tFA[] = {0.1f, 0.1f, 0.01f, 0.01f, 0.005f, 0.001f, 0.001f, 0.5f, 0.5f, 0.5f, 0.5f};
	float tMR = 0.09f;
	uint _tnumAPos[] = {20, 40, 80, 160, 320, 484, 484};

	for (uint i=0; i<6; i++)
	{
		tFA.push_back(_tFA[i]);
		numAPos.push_back(_tnumAPos[i]);
	}

	cvip::Adaboost a(pPath, nPath, valPath, numP, numN, tFA, tMR, numAPos, T);
	//cvip::MultiClassifier *m = a.train();
	
	std::vector<cvip::MultiClassifier*> cascades;

	cvip::MultiClassifier *frontal = new cvip::MultiClassifier(std::string("C:/Users/Cihan/Desktop/Face Detection Test Images/MCTCascade2_a.txt"));
	cvip::MultiClassifier *profileLeft = new cvip::MultiClassifier(std::string("C:/Users/Cihan/Desktop/Face Detection Test Images/MCTCascadeLeftProfile.txt"));
	cvip::MultiClassifier *profileRight = new cvip::MultiClassifier(std::string("C:/Users/Cihan/Desktop/Face Detection Test Images/MCTCascadeRightProfile.txt"));
	//cvip::MultiClassifier *profileRight->writeToFile("C:/Users/Cihan/Desktop/Face Detection Test Images/MCTProfil2.txt");
	//cvip::MultiClassifier *profileRight = profileLeft->getFlipped();
	cascades.push_back(frontal);
	//cascades.push_back(profileLeft);
	//cascades.push_back(profileRight);
	
	cvip::Tracker t(new cvip::FaceDetector(cascades));

	t.onVideo();
	
	//d.on_single_image(std::string("D:/Training Data/Negatives/0004.png"),0,0,false);
	//d.on_single_image(std::string("D:/Training Data/Positives/150.png"),0,0,false);
	//d.on_single_image(std::string("C:/Users/Cihan/Desktop/Face Detection Test Images/angola.png"),0,0,true);
	//d.on_single_image(std::string("C:/Users/Cihan/Desktop/MIT+CMU_Profil_Test/bbo-smith_r.png"),0,0,false);
	//d.on_video(0,true);
	//d.on_multiple_images("C:/Users/Cihan/Desktop/MIT+CMU_Test",0,0,true,true,false);

	char x;
	std::cin >> x;
	return 0;
}

