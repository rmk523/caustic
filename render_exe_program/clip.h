typedef float ScalarType;
#define max_edge_num 50

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

__device__ void grad_intersection( ScalarType* s, ScalarType * s_grad, ScalarType* e, ScalarType* e_grad, ScalarType *clipsquare,
     const int& flag, ScalarType* intersect, ScalarType* intersect_grad)
{
    // Vector2 / Jet2d
    switch(flag)
    {
        ScalarType quan; 
        case 0:
            intersect[0] = (s[0] - e[0]) * (clipsquare[1] - e[1]) / (s[1] - e[1]) + e[0];
            intersect[1] = ScalarType(clipsquare[1]);
            for(int i=0;i<6;i++)
            {   
                quan = (clipsquare[1] - e[1]) / (s[1] - e[1]);
                intersect_grad[i] = quan*s_grad[i] + (1-quan)*e_grad[i];
                intersect_grad[i] += (s[0] - e[0]) * (clipsquare[1] - e[1]) * (-1) / ((s[1] - e[1]) * (s[1] - e[1])) * s_grad[6+i];
                intersect_grad[i] += (s[0] - e[0]) * (clipsquare[1] - s[1]) / ((s[1] - e[1]) * (s[1] - e[1])) * e_grad[6+i];
                intersect_grad[6+i] = 0;
            }
            break;
        case 1:
            intersect[1] = (s[1] - e[1]) * (clipsquare[2] - e[0]) / (s[0] - e[0]) + e[1];
            intersect[0] = ScalarType(clipsquare[2]);
            for(int i=0;i<6;i++)
            {
                quan = (clipsquare[2] - e[0]) / (s[0] - e[0]);
                intersect_grad[6+i] = quan*s_grad[6+i] + (1-quan)*e_grad[6+i];
                intersect_grad[6+i] += (s[1] - e[1]) * (clipsquare[2] - e[0]) * (-1) / ((s[0] - e[0]) * (s[0] - e[0])) * s_grad[i];
                intersect_grad[6+i] += (s[1] - e[1]) * (clipsquare[2] - s[0]) / ((s[0] - e[0]) * (s[0] - e[0])) * e_grad[i];
                intersect_grad[i] = 0;
            }
            break;
        case 2:
            intersect[0] = (s[0] - e[0]) * (clipsquare[3] - e[1]) / (s[1] - e[1]) + e[0];
            intersect[1] = ScalarType(clipsquare[3]);
            for(int i=0;i<6;i++)
            {
                quan = (clipsquare[3] - e[1]) / (s[1] - e[1]);
                intersect_grad[i] = quan*s_grad[i] + (1-quan)*e_grad[i];
                intersect_grad[i] += (s[0] - e[0]) * (clipsquare[3] - e[1]) * (-1) / ((s[1] - e[1]) * (s[1] - e[1])) * s_grad[6+i];
                intersect_grad[i] += (s[0] - e[0]) * (clipsquare[3] - s[1]) / ((s[1] - e[1]) * (s[1] - e[1])) * e_grad[6+i];
                intersect_grad[6+i] = 0;
            }
            break;
        case 3:
            intersect[1] = (s[1] - e[1]) * (clipsquare[0] - e[0]) / (s[0] - e[0]) + e[1];
            intersect[0] = ScalarType(clipsquare[0]);
            for(int i=0;i<6;i++)
            {
                quan = (clipsquare[0] - e[0]) / (s[0] - e[0]);
                intersect_grad[6+i] = quan*s_grad[6+i] + (1-quan)*e_grad[6+i];
                intersect_grad[6+i] += (s[1] - e[1]) * (clipsquare[0] - e[0]) * (-1) / ((s[0] - e[0]) * (s[0] - e[0])) * s_grad[i];
                intersect_grad[6+i] += (s[1] - e[1]) * (clipsquare[0] - s[0]) / ((s[0] - e[0]) * (s[0] - e[0])) * e_grad[i];
                intersect_grad[i] = 0;
            }
            break;
        default:
            break;
    }
    // return intersect;
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

__device__ void grad_clip_Polygon(ScalarType* polygon, ScalarType* gradient, const int& length, ScalarType *clipsquare,
     ScalarType* newPolygon, ScalarType* newGradient, int& newLength)
{
    ScalarType inputPolygon[max_edge_num];
    ScalarType inputGradient[6*max_edge_num];
    int counter = 0;
    for(int i = 0; i < length; i ++)
    {
        for(int j = 0; j < 2; j++)
        {
            newPolygon[2*i+j] = polygon[2*i+j];
        }
        for(int j = 0; j < 12; j++)
        {
            newGradient[12*i+j] = gradient[12*i+j];
        }
    }
    newLength = length;

    for(int j = 0; j < 4; j++)
    {
        for(int k = 0; k < newLength; k++)
        {
            for(int l = 0; l < 2; l++)
            {
                inputPolygon[2*k+l] = newPolygon[2*k+l];
            }
            for(int l = 0; l < 12; l++)
            {
                inputGradient[12*k+l] = newGradient[12*k+l];
            }
        }
        counter = 0;
        for(int i = 0; i < newLength; i++)
        {
            int next_id = (i + 1) % newLength;
            ScalarType s[2] = {inputPolygon[2*i],inputPolygon[2*i+1]};
            ScalarType s_grad[12];
            for(int k=0; k<12; k++)
            {
                s_grad[k] = inputGradient[12*i+k];
            }
            ScalarType e[2] = {inputPolygon[2*next_id],inputPolygon[2*next_id+1]};
            ScalarType e_grad[12];
            for(int k=0; k<12; k++)
            {
                e_grad[k] = inputGradient[12*next_id+k];
            }
            ScalarType intersect[2];
            ScalarType intersect_grad[12];
            // const ScalarType& s = inputPolygon[i];
            // const ScalarType& e = inputPolygon[(i + 1) % newLength];
           
            if(inside(e, clipsquare, j))
            {
                if(inside(s, clipsquare, j))
                {
                    newPolygon[counter*2] = e[0];
                    newPolygon[counter*2+1] = e[1];
                    for(int k=0; k<12; k++)
                    {
                        newGradient[counter*12+k] = e_grad[k];
                    }
                    counter++;
                } 
                else
                {
                    grad_intersection(s, s_grad, e, e_grad, clipsquare, j, intersect, intersect_grad);
                    newPolygon[counter*2] = intersect[0];
                    newPolygon[counter*2+1] = intersect[1];
                    for(int k=0; k<12; k++)
                    {
                        newGradient[counter*12+k] = intersect_grad[k];
                    }
                    counter++;

                    newPolygon[counter*2] = e[0];
                    newPolygon[counter*2+1] = e[1];
                    for(int k=0; k<12; k++)
                    {
                        newGradient[counter*12+k] = e_grad[k];
                    }
                    counter++;
                } 
            }
            else if(inside(s, clipsquare, j))
            {
                grad_intersection(s, s_grad, e, e_grad, clipsquare, j, intersect, intersect_grad);
                newPolygon[counter*2] = intersect[0];
                newPolygon[counter*2+1] = intersect[1];
                for(int k=0; k<12; k++)
                {
                    newGradient[counter*12+k] = intersect_grad[k];
                }
                counter++;
            }
        }
        newLength = counter;
    }
}

__device__ ScalarType compute_area(ScalarType* new_poly, int new_poly_num)
{
    ScalarType area=0;
    for(int i=1;i<new_poly_num-1;i++)
    {
        area+=0.5*((new_poly[2*i]-new_poly[0])*(new_poly[2*i+3]-new_poly[1])-(new_poly[2*i+1]-new_poly[1])*(new_poly[2*i+2]-new_poly[0]));
    }
    return area;
}

__device__ ScalarType compute_grad_area(ScalarType* new_poly, ScalarType* new_poly_grad, int new_poly_num)
{
    ScalarType area=0;
    for(int i=1;i<new_poly_num-1;i++)
    {
        area+=0.5*((new_poly[2*i]-new_poly[0])*(new_poly[2*i+3]-new_poly[1])-(new_poly[2*i+1]-new_poly[1])*(new_poly[2*i+2]-new_poly[0]));
        
        new_poly_grad[2*i+1]+=0.5*(new_poly[0]-new_poly[2*i+2]);
        new_poly_grad[2*i+2]+=0.5*(new_poly[1]-new_poly[2*i+1]);
        new_poly_grad[2*i]+=0.5*(new_poly[2*i+3]-new_poly[1]);
        new_poly_grad[0]+=0.5*(new_poly[2*i+1]-new_poly[2*i+3]);
        new_poly_grad[2*i+3]+=0.5*(new_poly[2*i]-new_poly[0]);
        new_poly_grad[1]+=0.5*(new_poly[2*i+2]-new_poly[2*i]);
    }
    return area;
}

__device__ void accumulate_grad( ScalarType* small_area_grad, ScalarType* clip_poly_grad, ScalarType* new_poly_grad, int new_poly_num)
{
    for(int i = 0; i < 2 * new_poly_num; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            small_area_grad[j] += new_poly_grad[i] * clip_poly_grad[6*i+j];
        }
    }
}