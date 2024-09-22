`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    21:41:15 01/04/2022 
// Design Name: 
// Module Name:    Mux32to1 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module Mux32to1(
   output wire [31:0] Out,
	input  wire [1023:0] In,
	input  wire [4:0]  Sel
    );
   
	wire [127:0] muxconn;
	Mux8to1 mux81_1 (.Out(muxconn[31:0])  , .In(In[255:0])   , .Sel(Sel[2:0]));
	Mux8to1 mux81_2 (.Out(muxconn[63:32]) , .In(In[511:256]) , .Sel(Sel[2:0]));
	Mux8to1 mux81_3 (.Out(muxconn[95:64]) , .In(In[767:512]) , .Sel(Sel[2:0]));
	Mux8to1 mux81_4 (.Out(muxconn[127:96]), .In(In[1023:768]), .Sel(Sel[2:0]));
	Mux4to1 mux41 (.Out(Out), .In(muxconn), .Sel(Sel[4:3]));
	
endmodule
