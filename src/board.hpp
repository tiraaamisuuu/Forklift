#pragma once

#include "chess_types.hpp"
#include "legality_diagnostics.hpp"

struct Board {
    std::array<Piece, 64> b{};
    Color stm = Color::White;

    int epSquare = -1;          // en passant target square index or -1
    u8 castling = 0b1111;       // 1=WK,2=WQ,4=BK,8=BQ
    int halfmoveClock = 0;      // 50-move heuristic
    int fullmoveNumber = 1;
    std::array<int, 2> kingSquare{{-1, -1}};
    u64 hash = 0;

    const Zobrist* z = nullptr;

    void clear(){
        for(auto& p : b) p = Piece{};
        stm = Color::White;
        epSquare = -1;
        castling = 0b1111;
        halfmoveClock = 0;
        fullmoveNumber = 1;
        kingSquare = {{-1, -1}};
        hash = 0;
    }

    void reset(){
        clear();
        auto set = [&](int file, int rank, Color c, PieceType t){
            const int square = rank * 8 + file;
            b[static_cast<size_t>(square)] = Piece{t,c};
            if(t == PieceType::King) kingSquare[c == Color::White ? 0 : 1] = square;
        };

        // White
        set(0,0,Color::White,PieceType::Rook);
        set(1,0,Color::White,PieceType::Knight);
        set(2,0,Color::White,PieceType::Bishop);
        set(3,0,Color::White,PieceType::Queen);
        set(4,0,Color::White,PieceType::King);
        set(5,0,Color::White,PieceType::Bishop);
        set(6,0,Color::White,PieceType::Knight);
        set(7,0,Color::White,PieceType::Rook);
        for(int f=0; f<8; f++) set(f,1,Color::White,PieceType::Pawn);

        // Black
        set(0,7,Color::Black,PieceType::Rook);
        set(1,7,Color::Black,PieceType::Knight);
        set(2,7,Color::Black,PieceType::Bishop);
        set(3,7,Color::Black,PieceType::Queen);
        set(4,7,Color::Black,PieceType::King);
        set(5,7,Color::Black,PieceType::Bishop);
        set(6,7,Color::Black,PieceType::Knight);
        set(7,7,Color::Black,PieceType::Rook);
        for(int f=0; f<8; f++) set(f,6,Color::Black,PieceType::Pawn);

        stm = Color::White;
        epSquare = -1;
        castling = 0b1111;
        halfmoveClock=0;
        fullmoveNumber=1;

        recomputeHash();
    }

    bool loadFEN(const std::string& fen){
        clear();

        std::istringstream iss(fen);
        std::string boardPart, stmPart, castlingPart, epPart;
        int halfmove = 0;
        int fullmove = 1;
        if(!(iss >> boardPart >> stmPart >> castlingPart >> epPart >> halfmove >> fullmove)){
            return false;
        }

        int rank = 7;
        int file = 0;
        int whiteKings = 0;
        int blackKings = 0;
        for(char ch : boardPart){
            if(ch == '/'){
                if(file != 8) return false;
                rank--;
                file = 0;
                continue;
            }

            if(std::isdigit(static_cast<unsigned char>(ch))){
                int n = ch - '0';
                if(n < 1 || n > 8) return false;
                file += n;
                if(file > 8) return false;
                continue;
            }

            Color pc = std::isupper(static_cast<unsigned char>(ch)) ? Color::White : Color::Black;
            char lo = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            PieceType pt = PieceType::None;
            if(lo=='p') pt = PieceType::Pawn;
            else if(lo=='n') pt = PieceType::Knight;
            else if(lo=='b') pt = PieceType::Bishop;
            else if(lo=='r') pt = PieceType::Rook;
            else if(lo=='q') pt = PieceType::Queen;
            else if(lo=='k') pt = PieceType::King;
            else return false;

            if(rank < 0 || file >= 8) return false;
            b[rank*8 + file] = Piece{pt, pc};
            if(pt == PieceType::King){
                kingSquare[pc == Color::White ? 0 : 1] = rank * 8 + file;
                if(pc == Color::White) whiteKings++;
                else blackKings++;
            }
            file++;
        }

        if(rank != 0 || file != 8 || whiteKings != 1 || blackKings != 1) return false;

        if(stmPart == "w") stm = Color::White;
        else if(stmPart == "b") stm = Color::Black;
        else return false;

        castling = 0;
        if(castlingPart != "-"){
            for(char c : castlingPart){
                if(c=='K') castling |= 0b0001;
                else if(c=='Q') castling |= 0b0010;
                else if(c=='k') castling |= 0b0100;
                else if(c=='q') castling |= 0b1000;
                else return false;
            }
        }

        epSquare = -1;
        if(epPart != "-"){
            if(epPart.size() != 2) return false;
            char fch = epPart[0];
            char rch = epPart[1];
            if(fch < 'a' || fch > 'h' || rch < '1' || rch > '8') return false;
            int ef = int(fch - 'a');
            int er = int(rch - '1');
            epSquare = er*8 + ef;
        }

        halfmoveClock = std::max(0, halfmove);
        fullmoveNumber = std::max(1, fullmove);

        recomputeHash();
        return true;
    }

    Piece at(int idx) const { return b[idx]; }
    Piece& atRef(int idx) { return b[idx]; }

    void setZobrist(const Zobrist* zz){
        z = zz;
        recomputeHash();
    }

    bool hasEnPassantCapture() const {
        if(epSquare < 0) return false;
        const int targetRank = epSquare / 8;
        const int targetFile = epSquare % 8;
        const int pawnRank = targetRank + (stm == Color::White ? -1 : 1);
        if(pawnRank < 0 || pawnRank > 7) return false;

        for(const int fileOffset : {-1, 1}){
            const int pawnFile = targetFile + fileOffset;
            if(pawnFile < 0 || pawnFile > 7) continue;
            const Piece pawn = b[static_cast<size_t>(pawnRank * 8 + pawnFile)];
            if(pawn.t == PieceType::Pawn && pawn.c == stm) return true;
        }
        return false;
    }

    int enPassantHashFile() const {
        return hasEnPassantCapture() ? epSquare % 8 : 8;
    }

    void recomputeHash(){
        if(!z){ hash=0; return; }
        u64 h=0;
        for(int i=0;i<64;i++){
            Piece p=b[i];
            if(isNone(p)) continue;
            int c = (p.c==Color::White)?0:1;
            int pt = (int)p.t;
            h ^= z->psq[c][pt][i];
        }
        if(stm==Color::Black) h ^= z->sideToMove;
        h ^= z->castling[castling & 0xF];
        h ^= z->epFile[enPassantHashFile()];
        hash = h;
    }

    std::string toFEN() const {
        std::ostringstream out;
        for(int rank = 7; rank >= 0; rank--){
            int empty = 0;
            for(int file = 0; file < 8; file++){
                const Piece piece = b[static_cast<size_t>(rank * 8 + file)];
                if(isNone(piece)){
                    empty++;
                    continue;
                }
                if(empty > 0){ out << empty; empty = 0; }
                char symbol = '?';
                switch(piece.t){
                    case PieceType::Pawn: symbol = 'p'; break;
                    case PieceType::Knight: symbol = 'n'; break;
                    case PieceType::Bishop: symbol = 'b'; break;
                    case PieceType::Rook: symbol = 'r'; break;
                    case PieceType::Queen: symbol = 'q'; break;
                    case PieceType::King: symbol = 'k'; break;
                    default: break;
                }
                if(piece.c == Color::White){
                    symbol = static_cast<char>(std::toupper(static_cast<unsigned char>(symbol)));
                }
                out << symbol;
            }
            if(empty > 0) out << empty;
            if(rank > 0) out << '/';
        }

        out << (stm == Color::White ? " w " : " b ");
        if(castling == 0){
            out << '-';
        } else {
            if(castling & 0b0001) out << 'K';
            if(castling & 0b0010) out << 'Q';
            if(castling & 0b0100) out << 'k';
            if(castling & 0b1000) out << 'q';
        }
        out << ' ' << (epSquare >= 0 ? sqName(indexToSq(epSquare)) : "-")
            << ' ' << halfmoveClock << ' ' << fullmoveNumber;
        return out.str();
    }

    int findKing(Color c) const {
        return kingSquare[c == Color::White ? 0 : 1];
    }

    bool isSquareAttacked(int sq, Color by) const {
        int r = sq/8, f = sq%8;

        int pr = r + ((by==Color::White) ? -1 : 1);
        if(pr>=0 && pr<8){
            if(f-1>=0){
                Piece p=b[pr*8 + (f-1)];
                if(p.t==PieceType::Pawn && p.c==by) return true;
            }
            if(f+1<8){
                Piece p=b[pr*8 + (f+1)];
                if(p.t==PieceType::Pawn && p.c==by) return true;
            }
        }

        static const int kD[8][2]={{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
        for(auto& d: kD){
            int nf=f+d[0], nr=r+d[1];
            if(nf>=0&&nf<8&&nr>=0&&nr<8){
                Piece p=b[nr*8+nf];
                if(p.t==PieceType::Knight && p.c==by) return true;
            }
        }

        for(int df=-1; df<=1; df++){
            for(int dr=-1; dr<=1; dr++){
                if(df==0&&dr==0) continue;
                int nf=f+df, nr=r+dr;
                if(nf>=0&&nf<8&&nr>=0&&nr<8){
                    Piece p=b[nr*8+nf];
                    if(p.t==PieceType::King && p.c==by) return true;
                }
            }
        }

        auto ray = [&](int df, int dr, PieceType a, PieceType b2)->bool{
            int nf=f+df, nr=r+dr;
            while(nf>=0&&nf<8&&nr>=0&&nr<8){
                Piece p=this->b[nr*8+nf];
                if(!isNone(p)){
                    if(p.c==by && (p.t==a || p.t==b2)) return true;
                    return false;
                }
                nf+=df; nr+=dr;
            }
            return false;
        };

        if(ray(1,1,PieceType::Bishop,PieceType::Queen)) return true;
        if(ray(1,-1,PieceType::Bishop,PieceType::Queen)) return true;
        if(ray(-1,1,PieceType::Bishop,PieceType::Queen)) return true;
        if(ray(-1,-1,PieceType::Bishop,PieceType::Queen)) return true;

        if(ray(1,0,PieceType::Rook,PieceType::Queen)) return true;
        if(ray(-1,0,PieceType::Rook,PieceType::Queen)) return true;
        if(ray(0,1,PieceType::Rook,PieceType::Queen)) return true;
        if(ray(0,-1,PieceType::Rook,PieceType::Queen)) return true;

        return false;
    }

    bool inCheck(Color c) const {
        int k = findKing(c);
        if(k<0) return false;
        return isSquareAttacked(k, other(c));
    }

    template<typename MoveContainer>
    void genPseudoMoves(MoveContainer& out) const {
        out.clear();
        Color us = stm;

        auto push = [&](int from, int to, bool cap=false, bool ep=false, bool castle=false, PieceType promo=PieceType::None){
            Move m;
            m.from=(u8)from; m.to=(u8)to;
            m.isCapture=cap; m.isEnPassant=ep; m.isCastle=castle; m.promo=promo;
            out.push_back(m);
        };

        for(int i=0;i<64;i++){
            Piece p=b[i];
            if(isNone(p) || p.c!=us) continue;
            int r=i/8, f=i%8;

            if(p.t==PieceType::Pawn){
                int dir = (us==Color::White) ? 1 : -1;
                int startRank = (us==Color::White) ? 1 : 6;
                int promoRank = (us==Color::White) ? 7 : 0;

                int nr = r + dir;
                if(nr>=0 && nr<8){
                    int one = nr*8 + f;
                    if(isNone(b[one])){
                        if(nr==promoRank){
                            push(i, one, false, false, false, PieceType::Queen);
                            push(i, one, false, false, false, PieceType::Rook);
                            push(i, one, false, false, false, PieceType::Bishop);
                            push(i, one, false, false, false, PieceType::Knight);
                        } else {
                            push(i, one);
                            if(r==startRank){
                                int twoR = r + 2*dir;
                                int two = twoR*8 + f;
                                if(twoR>=0 && twoR<8 && isNone(b[two])) push(i, two);
                            }
                        }
                    }
                }

                for(int df : {-1,1}){
                    int nf=f+df, tr=r+dir;
                    if(nf<0||nf>=8||tr<0||tr>=8) continue;
                    int to=tr*8+nf;

                    if(!isNone(b[to]) && b[to].c!=us){
                        if(tr==promoRank){
                            push(i,to,true,false,false,PieceType::Queen);
                            push(i,to,true,false,false,PieceType::Rook);
                            push(i,to,true,false,false,PieceType::Bishop);
                            push(i,to,true,false,false,PieceType::Knight);
                        } else push(i,to,true);
                    }

                    if(epSquare==to){
                        int adj = r*8 + nf;
                        if(!isNone(b[adj]) && b[adj].t==PieceType::Pawn && b[adj].c!=us){
                            push(i,to,true,true,false);
                        }
                    }
                }

            } else if(p.t==PieceType::Knight){
                static const int d[8][2]={{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
                for(auto& dd: d){
                    int nf=f+dd[0], nr=r+dd[1];
                    if(nf<0||nf>=8||nr<0||nr>=8) continue;
                    int to=nr*8+nf;
                    if(isNone(b[to])) push(i,to);
                    else if(b[to].c!=us) push(i,to,true);
                }

            } else if(p.t==PieceType::Bishop || p.t==PieceType::Rook || p.t==PieceType::Queen){
                auto slide=[&](int df,int dr){
                    int nf=f+df, nr=r+dr;
                    while(nf>=0&&nf<8&&nr>=0&&nr<8){
                        int to=nr*8+nf;
                        if(isNone(b[to])) push(i,to);
                        else {
                            if(b[to].c!=us) push(i,to,true);
                            break;
                        }
                        nf+=df; nr+=dr;
                    }
                };

                if(p.t==PieceType::Bishop || p.t==PieceType::Queen){
                    slide(1,1); slide(1,-1); slide(-1,1); slide(-1,-1);
                }
                if(p.t==PieceType::Rook || p.t==PieceType::Queen){
                    slide(1,0); slide(-1,0); slide(0,1); slide(0,-1);
                }

            } else if(p.t==PieceType::King){
                for(int df=-1; df<=1; df++){
                    for(int dr=-1; dr<=1; dr++){
                        if(df==0&&dr==0) continue;
                        int nf=f+df, nr=r+dr;
                        if(nf<0||nf>=8||nr<0||nr>=8) continue;
                        int to=nr*8+nf;
                        if(isNone(b[to])) push(i,to);
                        else if(b[to].c!=us) push(i,to,true);
                    }
                }

                if(us==Color::White && i==4){
                    if((castling & 0b0001) && isNone(b[5]) && isNone(b[6]) &&
                       b[7].t==PieceType::Rook && b[7].c==Color::White){
                        if(!inCheck(Color::White) &&
                           !isSquareAttacked(5, Color::Black) &&
                           !isSquareAttacked(6, Color::Black))
                            push(4,6,false,false,true);
                    }
                    if((castling & 0b0010) && isNone(b[3]) && isNone(b[2]) && isNone(b[1]) &&
                       b[0].t==PieceType::Rook && b[0].c==Color::White){
                        if(!inCheck(Color::White) &&
                           !isSquareAttacked(3, Color::Black) &&
                           !isSquareAttacked(2, Color::Black))
                            push(4,2,false,false,true);
                    }
                }
                if(us==Color::Black && i==60){
                    if((castling & 0b0100) && isNone(b[61]) && isNone(b[62]) &&
                       b[63].t==PieceType::Rook && b[63].c==Color::Black){
                        if(!inCheck(Color::Black) &&
                           !isSquareAttacked(61, Color::White) &&
                           !isSquareAttacked(62, Color::White))
                            push(60,62,false,false,true);
                    }
                    if((castling & 0b1000) && isNone(b[59]) && isNone(b[58]) && isNone(b[57]) &&
                       b[56].t==PieceType::Rook && b[56].c==Color::Black){
                        if(!inCheck(Color::Black) &&
                           !isSquareAttacked(59, Color::White) &&
                           !isSquareAttacked(58, Color::White))
                            push(60,58,false,false,true);
                    }
                }
            }
        }
    }

    bool makeMove(const Move& m, Undo& u){
        if(m.from >= 64 || m.to >= 64) return false;
        u.m = m;
        u.epSquare = epSquare;
        u.castling = castling;
        u.halfmoveClock = halfmoveClock;
        u.fullmoveNumber = fullmoveNumber;
        u.kingSquare = kingSquare;
        u.hash = hash;
        u.captured = Piece{};

        Piece moving = b[m.from];
        if(isNone(moving)) return false;

        bool resetHalf = (moving.t==PieceType::Pawn) || m.isCapture || m.isEnPassant;
        halfmoveClock = resetHalf ? 0 : (halfmoveClock + 1);

        if(z){
            const int oldEpF = enPassantHashFile();
            hash ^= z->epFile[oldEpF];
            hash ^= z->castling[castling & 0xF];
            if(stm==Color::Black) hash ^= z->sideToMove;
        }

        epSquare = -1;

        if(m.isEnPassant){
            int dir = (moving.c==Color::White) ? -8 : 8;
            int capSq = int(m.to) + dir;
            u.captured = b[capSq];
            if(z && !isNone(u.captured)){
                int cc = (u.captured.c==Color::White)?0:1;
                hash ^= z->psq[cc][(int)u.captured.t][capSq];
            }
            b[capSq] = Piece{};
        } else if(m.isCapture){
            u.captured = b[m.to];
            if(z && !isNone(u.captured)){
                int cc = (u.captured.c==Color::White)?0:1;
                hash ^= z->psq[cc][(int)u.captured.t][(int)m.to];
            }
        }

        if(z){
            int mc = (moving.c==Color::White)?0:1;
            hash ^= z->psq[mc][(int)moving.t][(int)m.from];
        }

        b[m.to] = b[m.from];
        b[m.from] = Piece{};
        if(moving.t == PieceType::King){
            kingSquare[moving.c == Color::White ? 0 : 1] = m.to;
        }

        if(z){
            int mc = (moving.c==Color::White)?0:1;
            hash ^= z->psq[mc][(int)moving.t][(int)m.to];
        }

        if(m.promo != PieceType::None){
            if(z){
                int mc = (moving.c==Color::White)?0:1;
                hash ^= z->psq[mc][(int)PieceType::Pawn][(int)m.to];
                hash ^= z->psq[mc][(int)m.promo][(int)m.to];
            }
            b[m.to].t = m.promo;
        }

        if(m.isCastle){
            if(moving.c==Color::White){
                if(m.to==6){
                    Piece rook=b[7];
                    if(z){
                        int rc=0;
                        hash ^= z->psq[rc][(int)rook.t][7];
                        hash ^= z->psq[rc][(int)rook.t][5];
                    }
                    b[5]=b[7]; b[7]=Piece{};
                } else if(m.to==2){
                    Piece rook=b[0];
                    if(z){
                        int rc=0;
                        hash ^= z->psq[rc][(int)rook.t][0];
                        hash ^= z->psq[rc][(int)rook.t][3];
                    }
                    b[3]=b[0]; b[0]=Piece{};
                }
            } else {
                if(m.to==62){
                    Piece rook=b[63];
                    if(z){
                        int rc=1;
                        hash ^= z->psq[rc][(int)rook.t][63];
                        hash ^= z->psq[rc][(int)rook.t][61];
                    }
                    b[61]=b[63]; b[63]=Piece{};
                } else if(m.to==58){
                    Piece rook=b[56];
                    if(z){
                        int rc=1;
                        hash ^= z->psq[rc][(int)rook.t][56];
                        hash ^= z->psq[rc][(int)rook.t][59];
                    }
                    b[59]=b[56]; b[56]=Piece{};
                }
            }
        }

        auto clearIfTouches = [&](int sq, u8 mask){
            if(int(m.from)==sq || int(m.to)==sq) castling &= ~mask;
        };
        clearIfTouches(4,  0b0011);
        clearIfTouches(0,  0b0010);
        clearIfTouches(7,  0b0001);
        clearIfTouches(60, 0b1100);
        clearIfTouches(56, 0b1000);
        clearIfTouches(63, 0b0100);

        if(moving.t==PieceType::Pawn){
            int fr = int(m.from)/8;
            int tr = int(m.to)/8;
            if(std::abs(tr - fr) == 2){
                epSquare = (int(m.from) + int(m.to))/2;
            }
        }

        if(moving.c == Color::Black) fullmoveNumber++;
        stm = other(stm);

        if(inCheck(other(stm))){
            undoMove(u);
            return false;
        }

        if(z){
            const int newEpF = enPassantHashFile();
            hash ^= z->epFile[newEpF];
            hash ^= z->castling[castling & 0xF];
            if(stm==Color::Black) hash ^= z->sideToMove;
        }

        return true;
    }

    void undoMove(const Undo& u){
        const Move& m = u.m;

        stm = other(stm);

        epSquare = u.epSquare;
        castling = u.castling;
        halfmoveClock = u.halfmoveClock;
        fullmoveNumber = u.fullmoveNumber;
        kingSquare = u.kingSquare;
        hash = u.hash;

        Piece moved = b[m.to];

        if(m.isCastle){
            if(moved.c==Color::White){
                if(m.to==6){ b[7]=b[5]; b[5]=Piece{}; }
                else if(m.to==2){ b[0]=b[3]; b[3]=Piece{}; }
            } else {
                if(m.to==62){ b[63]=b[61]; b[61]=Piece{}; }
                else if(m.to==58){ b[56]=b[59]; b[59]=Piece{}; }
            }
        }

        b[m.from] = b[m.to];
        b[m.to] = Piece{};

        if(m.promo != PieceType::None){
            b[m.from].t = PieceType::Pawn;
        }

        if(m.isEnPassant){
            int dir = (b[m.from].c==Color::White) ? -8 : 8;
            int capSq = int(m.to) + dir;
            b[capSq] = u.captured;
        } else if(m.isCapture){
            b[m.to] = u.captured;
        }
    }

    void makeNullMove(NullUndo& u){
        u.epSquare = epSquare;
        u.hash = hash;

        if(z){
            hash ^= z->epFile[enPassantHashFile()];
            hash ^= z->epFile[8];
            hash ^= z->sideToMove;
        }
        epSquare = -1;
        stm = other(stm);
    }

    void undoNullMove(const NullUndo& u){
        stm = other(stm);
        epSquare = u.epSquare;
        hash = u.hash;
    }

    // Retain the original make/unmake legality filter as an independent oracle
    // and an optional benchmark build. Do not implement this in terms of the
    // optimised path: comparisons must be able to detect a missed pin/EP case.
    template<typename MoveContainer>
    void genLegalMovesReference(MoveContainer& legal){
#if defined(FORKLIFT_LEGALGEN_DIAGNOSTICS)
        LegalGenerationDiagnosticsScope diagnosticsScope;
#endif
        MoveList pseudo;
        genPseudoMoves(pseudo);
        legal.clear();
        legal.reserve(pseudo.size());

#if defined(FORKLIFT_LEGALGEN_DIAGNOSTICS)
        legalGenerationDiagnostics.pseudoCandidates += pseudo.size();
        // This extra check exists only in the instrumented reference build.
        // Preserve the original non-diagnostic filtering work unchanged.
        if((stm == Color::White || stm == Color::Black) &&
           findKing(stm) >= 0 && findKing(stm) < 64 && inCheck(stm)){
            legalGenerationDiagnostics.checkedCalls++;
        }
#endif
        for(const auto& m : pseudo){
#if defined(FORKLIFT_LEGALGEN_DIAGNOSTICS)
            legalGenerationDiagnostics.fullMakeUndoProbes++;
#endif
            Undo u{};
            if(makeMove(m,u)){
                legal.push_back(m);
                undoMove(u);
            }
        }
    }

    // Moving a non-king piece can expose its king only along a sliding ray.
    // Mark the first friendly blocker when a compatible enemy slider is the
    // next occupied square. Pinned moves still use the complete old oracle;
    // this mask is not an attempt to generate legal moves along the pin line.
    u64 potentiallyPinnedPieces(Color side, int king) const {
        static constexpr int directions[8][2] = {
            {1,1},{1,-1},{-1,1},{-1,-1},{1,0},{-1,0},{0,1},{0,-1}
        };
        const int kingFile = king % 8;
        const int kingRank = king / 8;
        u64 pinned = 0;
        for(int direction = 0; direction < 8; direction++){
            int file = kingFile + directions[direction][0];
            int rank = kingRank + directions[direction][1];
            int blocker = -1;
            while(file >= 0 && file < 8 && rank >= 0 && rank < 8){
                const int square = rank * 8 + file;
                const Piece piece = b[static_cast<size_t>(square)];
                if(!isNone(piece)){
                    if(blocker < 0){
                        if(piece.c != side) break;
                        blocker = square;
                    } else {
                        if(piece.c != side &&
                           (piece.t == PieceType::Queen ||
                            piece.t == (direction < 4 ? PieceType::Bishop : PieceType::Rook))){
                            pinned |= u64(1) << blocker;
                        }
                        break;
                    }
                }
                file += directions[direction][0];
                rank += directions[direction][1];
            }
        }
        return pinned;
    }

    template<typename MoveContainer>
    void genLegalMoves(MoveContainer& legal){
#if defined(FORKLIFT_REFERENCE_LEGALGEN)
        genLegalMovesReference(legal);
#else
        // Board is also used by tools/tests that can construct incomplete
        // positions directly. Only apply the fast-path proof to a recognised
        // side and consistent cached kings; preserve the old filter otherwise.
        const int whiteKing = kingSquare[0];
        const int blackKing = kingSquare[1];
        if((stm != Color::White && stm != Color::Black) ||
           whiteKing < 0 || whiteKing >= 64 || blackKing < 0 || blackKing >= 64 ||
           b[static_cast<size_t>(whiteKing)].t != PieceType::King ||
           b[static_cast<size_t>(whiteKing)].c != Color::White ||
           b[static_cast<size_t>(blackKing)].t != PieceType::King ||
           b[static_cast<size_t>(blackKing)].c != Color::Black){
            genLegalMovesReference(legal);
            return;
        }
#if defined(FORKLIFT_LEGALGEN_DIAGNOSTICS)
        LegalGenerationDiagnosticsScope diagnosticsScope;
#endif
        MoveList pseudo;
        genPseudoMoves(pseudo);
        legal.clear();
        legal.reserve(pseudo.size());

        const bool checked = inCheck(stm);
        const u64 pinned = checked ? 0 : potentiallyPinnedPieces(stm, findKing(stm));
#if defined(FORKLIFT_LEGALGEN_DIAGNOSTICS)
        legalGenerationDiagnostics.pseudoCandidates += pseudo.size();
        if(checked) legalGenerationDiagnostics.checkedCalls++;
#endif
        for(const Move& move : pseudo){
            // EP removes a second square; king moves change the attacked target.
            // Neither is covered by the single-origin unpinned-piece proof.
            // King-target captures on malformed positions retain the original
            // makeMove semantics rather than introducing a separate rules fix.
            if(!checked && b[move.from].t != PieceType::King &&
               !move.isEnPassant && !move.isCastle &&
               b[move.to].t != PieceType::King &&
               !(pinned & (u64(1) << move.from))){
                legal.push_back(move);
#if defined(FORKLIFT_LEGALGEN_DIAGNOSTICS)
                legalGenerationDiagnostics.fastAccepted++;
#endif
                continue;
            }
#if defined(FORKLIFT_LEGALGEN_DIAGNOSTICS)
            legalGenerationDiagnostics.fullMakeUndoProbes++;
#endif
            Undo undo{};
            if(makeMove(move, undo)){
                legal.push_back(move);
                undoMove(undo);
            }
        }
#endif
    }

    void genLegalMovesFrom(int from, std::vector<Move>& out){
        std::vector<Move> legal;
        genLegalMoves(legal);
        out.clear();
        for(const auto& m: legal) if(m.from==from) out.push_back(m);
    }

    bool insufficientMaterial() const {
        int wMinor=0,bMinor=0;
        int wB=0,wN=0,bB=0,bN=0;
        int whiteBishopSquare = -1;
        int blackBishopSquare = -1;
        for(size_t square = 0; square < b.size(); square++){
            const Piece& p = b[square];
            if(isNone(p) || p.t==PieceType::King) continue;
            if(p.t==PieceType::Pawn || p.t==PieceType::Rook || p.t==PieceType::Queen){
                return false;
            } else {
                if(p.c==Color::White){
                    wMinor++;
                    if(p.t==PieceType::Bishop){ wB++; whiteBishopSquare = static_cast<int>(square); }
                    if(p.t==PieceType::Knight) wN++;
                } else {
                    bMinor++;
                    if(p.t==PieceType::Bishop){ bB++; blackBishopSquare = static_cast<int>(square); }
                    if(p.t==PieceType::Knight) bN++;
                }
            }
        }
        if(wMinor==0 && bMinor==0) return true;
        if(wMinor==1 && bMinor==0 && (wB==1 || wN==1)) return true;
        if(bMinor==1 && wMinor==0 && (bB==1 || bN==1)) return true;
        if(wMinor==1 && bMinor==1 && wB==1 && bB==1){
            const auto squareColor = [](int square){ return ((square / 8) + (square % 8)) & 1; };
            return squareColor(whiteBishopSquare) == squareColor(blackBishopSquare);
        }
        return false;
    }
};

inline std::string moveToSAN(const Board& position, const Move& m){
    Piece moving = position.at(m.from);
    if(isNone(moving)) return moveToUCI(m);

    const Square fromSq = indexToSq(m.from);
    const Square toSq   = indexToSq(m.to);

    std::string san;
    if(m.isCastle){
        san = (toSq.file > fromSq.file) ? "O-O" : "O-O-O";
    } else {
        const bool isPawn = (moving.t == PieceType::Pawn);
        const bool isCapture = m.isCapture || m.isEnPassant;

        if(!isPawn){
            san.push_back(sanPieceChar(moving.t));

            Board probe = position;
            std::vector<Move> legal;
            probe.genLegalMoves(legal);

            bool ambiguous = false;
            bool sameFile = false;
            bool sameRank = false;
            for(const auto& cand : legal){
                if(cand.from == m.from || cand.to != m.to) continue;
                Piece cp = probe.at(cand.from);
                if(cp.t != moving.t || cp.c != moving.c) continue;
                ambiguous = true;
                if((cand.from % 8) == (m.from % 8)) sameFile = true;
                if((cand.from / 8) == (m.from / 8)) sameRank = true;
            }

            if(ambiguous){
                if(!sameFile){
                    san.push_back(char('a' + fromSq.file));
                } else if(!sameRank){
                    san.push_back(char('1' + fromSq.rank));
                } else {
                    san.push_back(char('a' + fromSq.file));
                    san.push_back(char('1' + fromSq.rank));
                }
            }
        }

        if(isPawn && isCapture){
            san.push_back(char('a' + fromSq.file));
        }
        if(isCapture){
            san.push_back('x');
        }

        san += sqName(toSq);

        if(m.promo != PieceType::None){
            san.push_back('=');
            san.push_back(sanPieceChar(m.promo));
        }
    }

    Board after = position;
    Undo u{};
    if(after.makeMove(m, u)){
        std::vector<Move> replies;
        after.genLegalMoves(replies);
        if(after.inCheck(after.stm)){
            san.push_back(replies.empty() ? '#' : '+');
        }
    }

    return san;
}
