`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    15:01:22 01/06/2022 
// Design Name: 
// Module Name:    ALUSTAGE 
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
module ALUSTAGE(
	input  wire [31:0] RF_A, RF_B, Immed,
	input  wire ALU_Bin_sel,
	input  wire [3:0] ALU_func,
	output wire [31:0] ALU_out,
	output wire ALU_Zero
    );

	wire [31:0] AluBin;
	
	Mux2to1 MuxBin(.Out(AluBin), .In({Immed, RF_B}), .Sel(ALU_Bin_sel));
	ALU AluMod(.Out(ALU_out), .Zero(ALU_Zero), .A(RF_A), .B(AluBin), .Op(ALU_func));

endmodule
