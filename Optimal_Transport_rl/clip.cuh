#include "power_diagram.hpp"
typedef double ScalarType;
#define max_edge_num 400

__device__ void egde_intersection( ScalarType* s,  ScalarType* e, ScalarType clipsquare, const int& flag, ScalarType* intersect)
{
    // Vector2 / Jet2d
    switch(flag)
    {
        case 0:
            intersect[0] = (s[0] - e[0]) * (clipsquare - e[1]) / (s[1] - e[1]) + e[0];
            intersect[1] = ScalarType(clipsquare);
            break;
        case 1:
            intersect[1] = (s[1] - e[1]) * (clipsquare - e[0]) / (s[0] - e[0]) + e[1];
            intersect[0] = ScalarType(clipsquare);
            break;
        case 2:
            intersect[0] = (s[0] - e[0]) * (clipsquare - e[1]) / (s[1] - e[1]) + e[0];
            intersect[1] = ScalarType(clipsquare);
            break;
        case 3:
            intersect[1] = (s[1] - e[1]) * (clipsquare - e[0]) / (s[0] - e[0]) + e[1];
            intersect[0] = ScalarType(clipsquare);
            break;
        default:
            break;
    }
    // return intersect;
}

__device__ void intersection( ScalarType* s,  ScalarType* e, ScalarType *clipsquare, const int& flag, ScalarType* intersect)
{
    // Vector2 / Jet2d
    switch(flag)
    {
        case 0:
            intersect[0] = (s[0] - e[0]) * (clipsquare[1] - e[1]) / (s[1] - e[1]) + e[0];
            intersect[1] = ScalarType(clipsquare[1]);
            break;
        case 1:
            intersect[1] = (s[1] - e[1]) * (clipsquare[2] - e[0]) / (s[0] - e[0]) + e[1];
            intersect[0] = ScalarType(clipsquare[2]);
            break;
        case 2:
            intersect[0] = (s[0] - e[0]) * (clipsquare[3] - e[1]) / (s[1] - e[1]) + e[0];
            intersect[1] = ScalarType(clipsquare[3]);
            break;
        case 3:
            intersect[1] = (s[1] - e[1]) * (clipsquare[0] - e[0]) / (s[0] - e[0]) + e[1];
            intersect[0] = ScalarType(clipsquare[0]);
            break;
        default:
            break;
    }
    // return intersect;
}

__device__ bool edge_inside( ScalarType* p_xy, ScalarType clipsquare, int& flag)
{
    switch(flag)
    {
        case 0:
            return (p_xy[1] > clipsquare);
        case 1:
            return (p_xy[0] < clipsquare);
        case 2:
            return (p_xy[1] < clipsquare);
        case 3:
            return (p_xy[0] > clipsquare);
        default:
            return 0;
    }
}

__device__ bool inside( ScalarType* p_xy, ScalarType *clipsquare, int& flag)
{
    switch(flag)
    {
        case 0:
            return (p_xy[1] > clipsquare[1]);
        case 1:
            return (p_xy[0] < clipsquare[2]);
        case 2:
            return (p_xy[1] < clipsquare[3]);
        case 3:
            return (p_xy[0] > clipsquare[0]);
        default:
            return 0;
    }
}

__device__ void clip_Polygon(ScalarType* polygon, const int& length, ScalarType *clipsquare, ScalarType* newPolygon, int& newLength)
{
    ScalarType inputPolygon[max_edge_num];
    int counter = 0;
    for(int i = 0; i < 2*length; i ++)
    {
        newPolygon[i] = polygon[i];
    }
    newLength = length;

    for(int j = 0; j < 4; j++)
    {
        for(int k = 0; k < 2*newLength; k++)
        {
            inputPolygon[k] = newPolygon[k];
        }
        counter = 0;
        for(int i = 0; i < newLength; i++)
        {
            int next_id = (i + 1) % newLength;
            ScalarType s[2] = {inputPolygon[2*i],inputPolygon[2*i+1]};
            ScalarType e[2] = {inputPolygon[2*next_id],inputPolygon[2*next_id+1]};
            ScalarType intersect[2];
            // const ScalarType& s = inputPolygon[i];
            // const ScalarType& e = inputPolygon[(i + 1) % newLength];
           
            if(inside(e, clipsquare, j))
            {
                if(inside(s, clipsquare, j))
                {
                    newPolygon[counter*2] = e[0];
                    newPolygon[counter*2+1] = e[1];
                    counter++;
                } 
                else
                {
                    intersection(s, e, clipsquare, j, intersect);
                    newPolygon[counter*2] = intersect[0];
                    newPolygon[counter*2+1] = intersect[1];
                    counter++;

                    newPolygon[counter*2] = e[0];
                    newPolygon[counter*2+1] = e[1];
                    counter++;
                } 
            }
            else if(inside(s, clipsquare, j))
            {
                intersection(s, e, clipsquare, j, intersect);
                newPolygon[counter*2] = intersect[0];
                newPolygon[counter*2+1] = intersect[1];
                counter++;
            }
        }
        newLength = counter;
    }
}

__device__ void edge_clip_Polygon(ScalarType* polygon, const int& length, ScalarType edge, int case_num, ScalarType* newPolygon, int& newLength)
{
    ScalarType inputPolygon[max_edge_num];
    int counter = 0;
    for(int i = 0; i < 2*length; i ++)
    {
        inputPolygon[i] = polygon[i];
    }
    newLength = length;
    int flag = (case_num + 3)%4;

    for(int i = 0; i < newLength; i++)
    {
        int next_id = (i + 1) % newLength;
        ScalarType s[2] = {inputPolygon[2*i],inputPolygon[2*i+1]};
        ScalarType e[2] = {inputPolygon[2*next_id],inputPolygon[2*next_id+1]};
        ScalarType intersect[2];

        if(edge_inside(e, edge, flag))
        {
            if(edge_inside(s, edge, flag))
            {
                newPolygon[counter*2] = e[0];
                newPolygon[counter*2+1] = e[1];
                counter++;
            } 
            else
            {
                egde_intersection( s, e, edge, flag, intersect);
                newPolygon[counter*2] = intersect[0];
                newPolygon[counter*2+1] = intersect[1];
                counter++;

                newPolygon[counter*2] = e[0];
                newPolygon[counter*2+1] = e[1];
                counter++;
            } 
        }
        else if(edge_inside(s, edge, flag))
        {
            egde_intersection( s, e, edge, flag, intersect);
            newPolygon[counter*2] = intersect[0];
            newPolygon[counter*2+1] = intersect[1];
            counter++;
        }
    }
    newLength = counter;

}
