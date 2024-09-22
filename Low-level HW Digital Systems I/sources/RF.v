`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    22:15:04 01/04/2022 
// Design Name: 
// Module Name:    RF 
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
module RF(
	output wire [31:0] Dout1, Dout2,
	input  wire [31:0] Din,
	input  wire [4:0] Ard1, Ard2, Awr,
	input  wire CLK, WrEn
    );
   
	wire [31:0] regSel; 
	wire [1023:0] regOut;
	
	Decoder5to32 dec532(.Out(regSel), .In(Awr), .E(1'b1));
	Register regFile[31:1](.Dout(regOut[1023:32]), .Data(Din), .CLK(CLK), .WE({31{WrEn}} & regSel[31:1]));
	Mux32to1 muxRead[1:0](.Out({Dout2, Dout1}), .In(regOut), .Sel({Ard2, Ard1}));
	
	// Ground pins corresponding to virtual 'register 0' so that its value cannot be changed and is always zero 
	assign regOut[31:0]  = 32'b0; 
	
endmodule
