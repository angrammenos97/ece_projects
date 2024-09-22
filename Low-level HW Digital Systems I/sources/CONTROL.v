`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    21:39:44 01/10/2022 
// Design Name: 
// Module Name:    CONTROL 
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
module CONTROL(
	//output wire [3:0] currentState,
	output wire PC_LdEn, PC_sel, RF_B_sel, RF_WrEn, RF_WrData_sel,
	            ALU_Bin_sel, MEM_we,
	output reg [3:0]	ALU_func, 
	output reg MEM_In_sel, MEM_Out_sel,
	input wire [31:0] Instr,
	input wire Clk, Reset, ALU_zero
    );
	
	// parameter defining the initial state of the porcessor's FSM
	parameter initialState = 3'b000;
	
	reg  [2:0] FSM_currentState, FSM_nextState; // nets to hold current and next FSM states respectively
	wire [5:0] opCode;                          // net to transmit the Instructions 6 MS Bits which correspond to the instruction's
                                               // opcode - used in the next_state_logic and output_logic blocks
	wire isnot_nop_instr;                       // wire to transmit whether an intsruction is a nop - see assignment below
	wire [31:0]Instr_buf;                       // net used as ouput of the 'mux21' instance that selects between Instr and initial
	                                            // value of 32'b0 (nop instruction) used during reset
	
	// mux used to select between input Instr signal and reset value
	Mux2to1 mux21(.Out(Instr_buf), .In({32'b0, Instr}), .Sel(Reset));
	
	always @(posedge Reset, posedge Clk)
		begin
			if (Reset)	
				begin: signals_reset
					//Reset FSM - initial FSM state is 0000
					FSM_currentState <= initialState; 
				end
			else			
				begin: next_state_assignment
					// if the Reset signal is inactive then assign new FSM state for next clock cycle according to next_state_logic block
					FSM_currentState <= FSM_nextState;	
				end				
		end
	
	always @(FSM_currentState, opCode, isnot_nop_instr)
		begin
			begin: next_state_logic
				// the state machine consists of 9 states a diagram of which can be found in FSM.png
				// synthesis with conditional case will result in the same netlist as if we had written the minterm explicitly
				FSM_nextState[0] = (((opCode[5:2] == 4'b1111) | (opCode[5:1] == 5'b00000)) & (FSM_currentState[1:0] == 2'b00)) |
				                   ((~opCode[5]) & isnot_nop_instr & (FSM_currentState[1:0] == 2'b01));
											
				FSM_nextState[1] = ((~opCode[5]) & isnot_nop_instr & (FSM_currentState[1:0] == 2'b01)) |
				                   (({FSM_currentState[2], FSM_currentState[0]} == 2'b00) & ((opCode[4:2] == 3'b011) | (opCode[4:1] == 4'b0001))) |
										 ((FSM_currentState[1:0] == 2'b00) & ({opCode[5], opCode[1:0]} == 3'b011));
										 
				
				FSM_nextState[2] = (({FSM_currentState[2], FSM_currentState[0]} == 2'b00) & ({opCode[5], opCode[2]} == 2'b10)) |
				                    ((FSM_currentState[2:1] == 2'b01) & ((opCode[4:2] == 3'b011) | (opCode[3:1] == 3'b001)));
										 
											
			end
			
		end
		
		
	always @(Instr_buf, opCode)
		begin: ALU_Func_logic
			// when the opCode starts with 2'b10 then the ALU_Func is determined by bits 5:0 of Instruction. When {opCode[5:4], opCode[1:0]} == 4'b1110
			// then the ALU should perform logical 'AND'. When {opCode[5:4], opCode[1:0]} == 4'b1111 the ALU should perform logical 'OR'. In all other cases
			// either the ALU operation is a dont care or it should perform addition. The above are implemented through a switch case
			casez(opCode)
				6'b10????: ALU_func = Instr_buf[3:0];
				
				6'b11??10: ALU_func = 4'b0010;
				
				6'b11??11: ALU_func = 4'b0011;
				
				6'b00000?: ALU_func = 4'b0001;
				
				default:   ALU_func = 4'b0000;
			endcase
	end
	
	always @(opCode, isnot_nop_instr)
		begin
			// ************************************ MEM_In_sel ***********************************************
			// the only cases where MEM_WrEn is '1' is when a store instruction is executed. As a result all 
			// other cases can be considered as don't care for simplicity. Based on a store instruction's opCode
			// and the connectivity of the MEM_In_mux21 in Processor.v MEM_In_sel = opCode[3] 
			MEM_In_sel = opCode[3] & isnot_nop_instr;
			
			
			// ************************************ MEM_Out_sel ***********************************************
			// the only cases where data is read from MEM is when a load instruction is executed. As a result all 
			// other cases can be considered as don't care for simplicity. Based on a load instruction's opCode
			// and the connectivity of the MEM_Out_mux21 in Processor.v MEM_Out_sel = opCode[2] 
			MEM_Out_sel = opCode[2] & isnot_nop_instr;
		end
		
	// all signals apart from Pc_LdEn are anded with the signal indicating whether this is an nop instruction in order to avoid their activation	
	begin: output_logic
			// ************************************ PC_LdEn logic ******************************************** 
			// since loading a new instruction takes two cycles (two FFs, one for the PC and one for the instruction memory), the 
			// signal is enabled two states prior to the initial state for every data path. For example if an ALU operation is performed
			// then the PC_LdEn is enable right at the begining when the FSM is in state 000 so that the correct PC is loaded, while the 
			// PC does not jump to the next instruction.
			assign PC_LdEn = ((FSM_currentState[2:0] == 3'b000) & ((opCode[4:2] != 3'b011) & (opCode[4:1] != 4'b0001)) & ((opCode[5:1] != 5'b00000) | (~isnot_nop_instr))) |
			                 ((FSM_currentState[2:0] == 3'b010) & ((opCode[4:2] == 3'b011) | (opCode[3:1] == 3'b0001))) |
								  ((FSM_currentState[2:0] == 3'b001) & (opCode[5:2] != 4'b1111) & isnot_nop_instr);
			
			// ************************************ PC_sel logic *********************************************
			// active only when a branch command is being executed and the condition for branching is true for 'beq' and false for 'bne'
			// while when command is 'b' then PC_sel is active regardless - no further opportunity for logic reduction was identified
			assign PC_sel = ((opCode[5:2] == 4'b1111) | (ALU_zero & (opCode == 6'b000000)) | ((~ALU_zero) & (opCode == 6'b000001))) & isnot_nop_instr;

			// ************************************ RF_B_sel logic *******************************************
			// implemented as the negation of the cases where the signal should be '0' as it resulted in less logic
			// Signal's logic doesn't take into consideration the state of the FSM but rather the opcode of the instruction.
			// More specifically, RF_B_sel = ~opCode[5] when the two operands should be read from rs and rt registers, while 
			// for all other opcodes either the rd register is read or nothing is read, which was interpreted as a don't care.
			assign RF_B_sel = ~opCode[5] & isnot_nop_instr;	
			
			// ************************************ RF_WrData_sel logic *******************************************
			// implemented as the negation of the cases where the signal should be '0'. Since controlling the write
			// function in the RF is done through the RF_WrEn, RF_WrData_sel could be anything in cases where RF_WrEn
			// is inactive. As a result, these cases where considered as a don't care in order to reduce the combinational
			// logic involved. Thus, since the signal should be '0' for all ALU operations and '1' for load operations
         // the resulting logic is the same as in RF_B_sel. It should be noted that for li and lui operations, since
         // bits [25-20] are zero, are implemented as an ALU operation where the 'zero register' is added to the Immed
         // value.
         assign RF_WrData_sel = ~opCode[5] & isnot_nop_instr;		


         // ************************************* RF_WrEn logic *******************************************	
         // Write to the RF should be performed only when the data coming from the write data mux are valid. Thus
			// only when the operation has been completed. In the current FSM this condition is true one cycle prior
			// the initial state of the FSM and only for instructions that are not store or branch. Hence this condition 
			// applies only to states '110' and '100'
         assign RF_WrEn = FSM_currentState[2] & (~FSM_currentState[0]) & isnot_nop_instr; 


         // ************************************* ALU_Bin_sel *********************************************
			// Signal should be active only when the immediate is added to the rs register and stored in RF or is used as an 
			// address to access the memory. The rest of the operations Rf_B is used or the ALU is not used at all(thus the state
			// considered as a don't care). The resultiing logic is based on the opCode and can be seen below
         assign ALU_Bin_sel = ((opCode[5] & opCode[4]) | (opCode[1] & opCode[0])) & isnot_nop_instr;	


         // ************************************* MEM_we **************************************************
			// signal should be active only when a store operation is performed and data are ready to be written
			// in memory. Thus signal is '1' only when FSM_currentState == 010 & FSM_nextState == 000
			assign MEM_we = (FSM_currentState[2:0] == 3'b010) & ({FSM_nextState[2], FSM_nextState[0]} == 2'b00) & isnot_nop_instr;
	
	end
	
	assign opCode          = Instr_buf[31:26];
	assign isnot_nop_instr = |(Instr[31:0]);  // identify whether this instruction is a nope ('0' if it is) 

endmodule
