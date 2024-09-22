/*
 * Library containing function implementing addition of two
 * half-precision numbers in C.
 *
 * Author
 * ===================================================================
 * (name)                                       (email)
 *
 * AUTh, (date)
 */

#include "half.h"

/*
 * addhalf: Add two half-precision floating point numbers using
 * integer operations.
 *
 * INPUTS
 *   a: address of first number in half
 *   b: address of second number in half
 *
 * OUTPUT
 *   sum: The result of the addition of the two numbers in half.
 *
 * NOTE
 *   This function only implements base case scenario. It does not
 *   take into account corner cases. It should include the cases when
 *   the output is equal to zero or both inputs are equal to zero.
 *
 *   Validate following inputs:
 *      0.0 +  0.0 =  0.0
 *      1.0 + -1.0 =  0.0
 *      0.1 +  0.0 =  0.1
 *     -0.1 +  0.0 = -0.1
 */
half addhalf( const half a, const half b )
{

    half sum = 0;

    /* <<YOUR-CODE-HERE>> */
    int sa = a & 0x8000;         /* sign a*/
    int ea = (a & 0x7c00) >> 10; /* exponent a*/
    int ma = a & 0x03ff;         /* mantissa a*/

    int sb = b & 0x8000;         /* sign b*/
    int eb = (b & 0x7c00) >> 10; /* exponent b*/
    int mb = b & 0x03ff;         /* mantissa b*/

    /*/printf("LaaaaaL a> %d %d %d\n",sa,ea,ma);
    //printf("LaaaaaL b> %d %d %d\n",sb,eb,mb);*/

    int s=0,e=0,m=0;

    ma+=1024;
    mb+=1024;
    unsigned int aorb=0;           /*0 for a>b and 1 for a<b*/
    if (ea==eb){
        if ((ma==1024) && (mb==1024))
            m=0;
        else if(ma>mb)
            aorb=0;
        else
            aorb=1;
        e=ea;
    }

    else{
        if((ea==0)&&(ma==1024)){
            s=sb;
            e=eb;
            m=mb;
            goto NORM;
        }
        else if((eb==0)&&(mb==1024)){
            s=sa;
            e=ea;
            m=ma;
            goto NORM;
        }
        else{
            while(ea!=eb){
                if(ea<eb){
                    aorb=1;
                    s=sb;
                    ea++;
                    ma=ma/2;
                }
                else{
                    aorb=0;
                    s=sa;
                    eb++;
                    mb=mb/2;
                }
            }
            e=ea;
        }
    }

    if(sa==sb){
            s=sa;
            m=ma+mb;
        }
        else{
            if(!aorb){
                s=sa;
                m=ma-mb;
            }
            else{
                s=sb;
                m=mb-ma;
            }
        }

    NORM:
    /*/printf("LaaaaaaL sum> %d %d %d\n",s,e,m);*/
    if (((e==0)&&(m==2048))|| m==0){
        sum=s;
        /*/printf("ola mhden\n");*/
    }
    else{
        if(m>=1024){
            m-=1024;
            while(m>=1024){
                m-=1024;
                e++;
            }
        }
        else{
            m=m*2;
            e--;
        }

        sum= s | e  << 10 | m;
    }


    /*/printf("LOOOOOL a> %d %d %d\n",sa,ea,ma);
    //printf("LOOOOOL b> %d %d %d\n",sb,eb,mb);
    //printf("LOOOOOL sum> %d %d %d",s,e,m);*/

    return sum;

}
