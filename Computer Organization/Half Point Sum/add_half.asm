.text

############################################################
###     Enter your code here.
###     Write function addhalf

# addhalf -- Add two half-precision floating point numbers using
# integer operations.
#
# Inputs:
#       $a0:    1st FP16 (half)
#       $a1:    2nd FP16 (half)
# Outputs:
#       $v0:    $a0 + $a1 in FP16 (half) format
#
# NOTE
#   This function only implements base case scenario. It does not
#   take into account corner cases. It should include the cases when
#   the output is equal to zero or both inputs are equal to zero.
#
#   Validate following inputs:
#      0.0 +  0.0 =  0.0
#      1.0 + -1.0 =  0.0
#      0.1 +  0.0 =  0.1
#     -0.1 +  0.0 = -0.1
#
addhalf:
	## <<YOUR-CODE-HERE>>
  ## prepare sgns
  andi $t0, $a0, 0x8000   # sa on t0
  andi $t1, $a1, 0x8000   # sb on t1
  add $t6, $zero, $zero   # s on t6

  ## prepare exps
  andi $t2, $a0, 0x7c00
  srl $t2, $t2, 10        # ea on t2
  andi $t3, $a1, 0x7c00
  srl $t3, $t3, 10        # eb on t3
  add $t7, $zero, $zero   # e on t7

  ## prepare mantissas
  andi $t4, $a0, 0x03ff   # ma on t4
  andi $t5, $a1, 0x03ff   # mb on t5
  add $t8, $zero, $zero   # m on t8
  add $v0, $v0, $zero     # v0=0

  addi $t4, $t4, 1024     # ma+=1024
  addi $t5, $t5, 1024     #mb+=1024
  add $t9, $zero $zero    # 0 for a>b and 1 for a<b

  bne $t2, $t3, eaNoteb  # if ea != eb then eaNoteb
  add $t7, $t2, $zero   #e=ea=eb
  bne $t4, 1024, signsCheck # if ma=mb=1024 m=0
  bne $t5, 1024, signsCheck #else continue sign
  add $t8, $zero, $zero
  blt $t4, $t5, maLessmb  # if  ma<mb  then t9=1
  j signsCheck

maLessmb:
  addi $t9, $zero, 1      # t9 turn to 1 (a<b)
  j signsCheck

eaNoteb:
  bne $t2, $zero, ebCheck  # check if a=0
  bne $t4, 1024, ebCheck
  add $t6, $t1, $zero   # s=sb
  add $t7, $t3, $zero   # e=eb
  add $t8, $t5, $zero   # m=mb
  j NORM
ebCheck:                # check if b=0
  bne $t3, $zero, matching
  bne $t5, 1024, matching
  add $t6, $t0, $zero   # s=sa
  add $t7, $t2, $zero   # e=ea
  add $t8, $t4, $zero   # m=ma
  j NORM
matching:
L1:
  beq $t2, $t3, ExitL1  # while ea!=eb
  bgt $t2, $t3, eaGreateb   # if ea<eb do.. else..
  addi $t9, $zero, 1    # turn t9 to 1 (a<b)
  add $t6, $t1, $zero   # s=sb
  addi $t2, $t2, 1      # ea++
  srl $t4, $t4, 1       # ma=ma/2
  j L1
eaGreateb:              # else ea>eb
  add $t9, $zero, $zero # turn t9 to 0 (a>b)
  add $t6, $t0, $zero   # s=sa
  addi $t3, $t3, 1      # eb++
  srl $t5, $t5, 1       # mb=mb/2
  j L1
ExitL1:
  add $t7, $t2, $zero   #e=ea

signsCheck:
  bne $t0, $t1 notSimsigns     # if sa=sb do.. else..
  add $t6, $t0, $zero   # s=sa
  add $t8, $t4, $t5     # m=ma+mb
  j NORM
notSimsigns:
  beq $t9, 1, aLessb
  add $t6, $t0, $zero   # s=sa
  sub $t8, $t4, $t5     # m=ma-mb
  j NORM
aLessb:
  add $t6, $t1, $zero   # s=sb
  sub $t8, $t5, $t4     # m=mb-ma

NORM:
  bne $t7, $zero, checkZero # e=0 & m=2048
  bne $t8, 2048, checkZero
  add $v0, $t6, $zero        # v0=+-0
  jr $ra
checkZero:
  bne $t8, $zero, notZero # m=0
  add $v0, $t6, $zero     # v0=+-0
  jr $ra
notZero:
  blt $t8, 1024, tooBig   # if m>=1024
  bge $t8, 2048, tooBig   # if m<2048
  sub $t8, $t8, 1024      # m-=1024
  j End
tooBig:
  blt $t8, 2048, mLess    # if m>=2048 do loop
L2:
  blt $t8, 2048, ExitL2   # while m>=2048
  srl $t8, $t8, 1         # m=m/2
  addi $t7, $t7, 1        # e++
  j L2
mLess:
  sll $t8, $t8, 1     # m*=2
  addi $t7, $t7, -1   # e--
  j End
ExitL2:
  sub $t8, $t8, 1024  # m-=1024
End:
  sll $t7, $t7, 10    # e << 10
  or $v0, $t6, $t7    # result = s | e
  or $v0, $v0, $t8    #                |m
  jr $ra
###     addhalf ending
############################################################


############################################################
###     DO NOT CHANGE ANYTHING BELOW !!!


.data

strInputFirst:
        .asciiz "Input first floating point number: "

strInputSecond:
        .asciiz "Input second floating point number: "

strResult:
        .asciiz "Result: "

.text

        .globl  main
main:

        # request first floating point number
        la      $a0, strInputFirst
        jal     printString
        li      $v0, 6
        syscall

        # move number to integer register s0
        mfc1    $s0, $f0

        # request second floating point number
        la      $a0, strInputSecond
        jal     printString
        li      $v0, 6
        syscall

        # move number to integer register s1
        mfc1    $s1, $f0

        # transform 1st float to half
        move    $a0, $s0
        jal     float2half
        move    $s0, $v0

        # transform 2nd float to half
        move    $a0, $s1
        jal     float2half
        move    $s1, $v0

        # add 2 half numbers
        move    $a0, $s0
        move    $a1, $s1
        jal     addhalf

        # transform to full float
        move    $a0, $v0
        jal     half2float

        # print result
        mtc1    $v0, $f12
        la      $a0, strResult
        jal     printString
        jal     printFloatln

        # exit
        li      $v0, 10
        syscall


# float2half -- Convert a full precision floating point to half
# precision
#
# Inputs:
#       $a0:    FP32 number
# Outputs:
#       $v0:    FP16 number
#
float2half:
        ## prepare exp
        srl     $t0, $a0, 23    # e = a >> 23
        andi    $t0, $t0, 0xff  # e = e & 0xff
        addi    $t0, $t0, -127  # e -= 127 -- unbias
        addi    $t0, $t0, 15    # e += 15  -- rebias
        sll     $t0, $t0, 10    # e << 10

        ## prepare sgn
        srl     $t1, $a0, 16     # s = a >> 16
        andi    $t1, $t1, 0x8000 # s = s && 0x8000

        ## prepare mantissa
        srl     $t2, $a0, 13     # m = a >> 13
        andi    $t2, $t2, 0x03ff # m = a & 0x03ff

        ## check if zero, otherwise transform

        slti    $t6, $t0, -15         # if e < -15
        beqz    $t6, float2half_build #
        move    $v0, $0               #   result = 0
        jr      $ra                   #   return result
                                      #
float2half_build:                     # else
        or      $v0, $t0, $t1         #   result = e | s
        or      $v0, $v0, $t2         #                  | m
                                      # endif
        jr      $ra                   # return result


# half2float -- Convert a half precision floating point to full
# precision
#
# Inputs:
#       $a0:    FP16 number
# Outputs:
#       $v0:    FP32 number
#
half2float:
        # prepare exp
        andi    $t0, $a0, 0x7c00 # e = a & 0x7c00
        srl     $t0, $t0, 10     # e = e >> 10

        # prepare sgn
        andi    $t1, $a0, 0x8000 # s = a & 0x8000
        sll     $t1, $t1, 16     # s <<= 16

        # prepare mantissa
        andi    $t2, $a0, 0x03ff # m = a & 0x03ff
        sll     $t2, $t2, 13     # m <<= 13

        ## check if zero, otherwise transform
        beqz    $t0, res_0            # if e > 0
        addi    $t0, $t0, 127         #   e += 127 -- unbias
        addi    $t0, $t0, -15         #   e -= 15  -- rebias
        sll     $t0, $t0, 23          #   e <<= 23
        or      $v0, $t0, $t1         #   result = e | s
        or      $v0, $v0, $t2         #                  | m
        jr      $ra                   #   return result
res_0:                                # else
        move    $v0, $0               #   result = 0
        jr      $ra                   #   return result
                                      # endif

# printString -- Print input string to console
#
# Inputs:
#       $a0:    Memory address of string
#
# Outputs:
#       (none)
#
printString:

        # print input string
        addi    $v0, $0, 4
        syscall

        jr      $ra


# printFloat -- Print input float to console, followed by new line
#
# Inputs:
#       $f12:    Float value
#
# Outputs:
#       (none)
#
printFloatln:

        # print float
        li      $v0, 2
        syscall

        # print new line
        addi    $v0, $0, 11     # ASCII character print
        li      $a0, 10         # ASCII character new line
        syscall

        jr      $ra
